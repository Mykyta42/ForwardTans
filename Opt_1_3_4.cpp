#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include "defines.h"
#include "Opt_1_3_4.h"
using namespace std;


namespace HuffEncodeV2_1_1
{
	void fillHuffLens(vector <array<int32_t, 2> >& cntPar, vector <uint32_t>& huffLens, uint32_t dictSize,
		int32_t i, uint32_t depth)
	{
		if (i < dictSize)
		{
			huffLens[i] = depth;
			return;
		}

		fillHuffLens(cntPar, huffLens, dictSize, cntPar[i][0], depth + 1);
		fillHuffLens(cntPar, huffLens, dictSize, cntPar[i][1], depth + 1);
	}

	void huffEncode(uint32_t* codes, uint32_t cntCodes, uint32_t dictSize, uint32_t* stream,
		uint32_t& streamLen, uint8_t* huffTable, uint32_t& huffTableLen)
	{
		vector <array<int32_t, 2> > cntPar(2 * dictSize - 1); // weight, parent
		for (auto& it : cntPar) { it[0] = 0; it[1] = -1; }

		// count frequencies
		for (int32_t i = 0; i < cntCodes; i++)
		{
			cntPar[codes[i]][0]++;
		}

		double entropyBits = 0;
		for (int32_t i = 0; i < dictSize; i++)
		{
			entropyBits -= cntPar[i][0] * log2(double(cntPar[i][0]) / cntCodes);
		}
		cout << "Entropy bytes: " << int64_t(entropyBits / 8) << "\n";

		// build some tree to get huffman codes lengths
		int32_t iLeaf = dictSize - 1, iNode = dictSize, nextNode = dictSize;

		for (int32_t iter = 0; iter < dictSize - 1; iter++)
		{
			if (iter == 0 || (iLeaf > 1 && cntPar[iLeaf - 1][0] <
				cntPar[iNode][0]))
			{
				cntPar[nextNode][0] = cntPar[iLeaf][0] + cntPar[iLeaf - 1][0];
				cntPar[iLeaf][1] = nextNode;
				cntPar[iLeaf - 1][1] = nextNode;
				iLeaf -= 2;
				nextNode++;
			}
			else if (iLeaf == -1 || (abs(iNode - nextNode) > 1 &&
				cntPar[iNode + 1][0] < cntPar[iLeaf][0]))
			{
				cntPar[nextNode][0] = cntPar[iNode][0] + cntPar[iNode + 1][0];
				cntPar[iNode][1] = nextNode;
				cntPar[iNode + 1][1] = nextNode;
				iNode += 2;
				nextNode++;
			}
			else
			{
				cntPar[nextNode][0] = cntPar[iLeaf][0] + cntPar[iNode][0];
				cntPar[iLeaf][1] = nextNode;
				cntPar[iNode][1] = nextNode;
				iLeaf--;
				iNode++;
				nextNode++;
			}
		}

		// now cntPar is represented as {child 1, child 2}
		for (auto& it : cntPar) { it[0] = -1; }
		for (int32_t i = int32_t(cntPar.size()) - 2; i >= 0; i--)
		{
			int32_t parent = cntPar[i][1];

			if (cntPar[parent][0] == -1)
			{
				cntPar[parent][0] = i;
			}
			else
			{
				cntPar[parent][1] = i;
			}
		}

		// prepare canonical huffman codes arrays
		vector <uint32_t> huffLens(dictSize);
		vector <uint32_t> huffCodes(dictSize);

		fillHuffLens(cntPar, huffLens, dictSize, int32_t(cntPar.size()) - 1, 0);

		uint32_t minHuffCodeLen = huffLens[0];
		uint32_t maxHuffCodeLen = huffLens[dictSize - 1];

		vector <int32_t> cntCodesPerLen(32, 0);
		for (int32_t i = 0; i < dictSize; i++)
		{
			cntCodesPerLen[huffLens[i]]++;
		}

		// make codes
		huffCodes[0] = 0;
		for (int32_t i = 1; i < dictSize; i++)
		{
			huffCodes[i] = (huffCodes[i - 1] + 1) << (huffLens[i] - huffLens[i - 1]);
		}

		cout << "Min/Max huff code len: " << minHuffCodeLen << " " << maxHuffCodeLen << "\n";

		// encode
		uint64_t buffer = 0;
		uint32_t bufferShift = 64;
		streamLen = 0;

		for (int32_t i = 0; i < cntCodes; i++)
		{
			uint32_t code = codes[i];
			uint32_t codeLen = huffLens[code];
			uint64_t huffCode = huffCodes[code];
			bufferShift -= codeLen;
			buffer |= huffCode << bufferShift;

			if (bufferShift <= 32)
			{
				stream[streamLen] = buffer >> 32;
				buffer <<= 32;
				bufferShift += 32;
				streamLen++;
			}
		}

		stream[streamLen] = buffer >> 32;
		stream[streamLen + 1] = buffer & 0xFFFFFFFF;
		stream[streamLen + 2] = 0;
		stream[streamLen + 3] = 0;
		stream[streamLen + 4] = 0;
		stream[streamLen + 5] = 0;
		stream[streamLen + 6] = 0;
		stream[streamLen + 7] = 0;
		streamLen += 8;

		for (int32_t i = 0; i + 1 < streamLen; i += 2)
		{
			swap(stream[i], stream[i + 1]);
		}

		huffTableLen = 0;
		*(uint32_t*)(huffTable + huffTableLen) = minHuffCodeLen;
		huffTableLen += 4;
		*(uint32_t*)(huffTable + huffTableLen) = maxHuffCodeLen;
		huffTableLen += 4;
		for (int32_t i = minHuffCodeLen; i <= maxHuffCodeLen; i++)
		{
			*(uint32_t*)(huffTable + huffTableLen) = cntCodesPerLen[i];
			huffTableLen += 4;
		}
	}
}

#define CODES_PER_BITSTREAM 3
#define CODES_BLOCK_SIZE (CODES_PER_BITSTREAM * 64u)

#define LOOKUP_BITS 12
#define LOOKUP_SIZE (1 << LOOKUP_BITS)
#define LOOKUP_MASK (LOOKUP_SIZE - 1)

namespace HuffDecodeV2_1_1
{
	uint32_t t_minHuffLen, t_maxHuffLen;
	uint32_t t_cntCodesPerLen[32];
	uint32_t t_firstCode[32], t_firstHuffCode[32];
	uint32_t t_firstCodeDiff[32]; // = t_firstCode[i] - t_firstHuffCode[i] (2 operation -> 1 operation)
	uint64_t t_limit[32];

#if HUFF_TYPE == 2'1'1
	int8_t t_first[LOOKUP_SIZE];
#else
	int8_t t_first[2];
#endif

	void precompute(uint8_t* huffTable, uint32_t huffTableLen, uint32_t dictSize)
	{
		int32_t huffTablePos = 0;

		t_minHuffLen = *(uint32_t*)(huffTable + huffTablePos);
		huffTablePos += 4;
		t_maxHuffLen = *(uint32_t*)(huffTable + huffTablePos);
		huffTablePos += 4;
		for (int32_t i = t_minHuffLen; i <= t_maxHuffLen; i++)
		{
			t_cntCodesPerLen[i] = *(uint32_t*)(huffTable + huffTablePos);
			huffTablePos += 4;
		}

		int32_t lastHuffCode, lastHuffCodeLen;

		t_firstCode[t_minHuffLen] = 0;
		t_firstHuffCode[t_minHuffLen] = 0;
		t_firstCodeDiff[t_minHuffLen] = t_firstCode[t_minHuffLen] - t_firstHuffCode[t_minHuffLen];
		lastHuffCode = t_firstHuffCode[t_minHuffLen] + t_cntCodesPerLen[t_minHuffLen] - 1;
		lastHuffCodeLen = t_minHuffLen;

		for (int32_t i = t_minHuffLen + 1; i <= t_maxHuffLen; i++)
		{
			if (t_cntCodesPerLen[i] > 0)
			{
				t_firstCode[i] = t_firstCode[lastHuffCodeLen] + t_cntCodesPerLen[lastHuffCodeLen];
				t_firstHuffCode[i] = (lastHuffCode + 1) << (i - lastHuffCodeLen);
				t_firstCodeDiff[i] = t_firstCode[i] - t_firstHuffCode[i];
				lastHuffCode = t_firstHuffCode[i] + t_cntCodesPerLen[i] - 1;
				lastHuffCodeLen = i;
			}
			else
			{
				t_firstCode[i] = t_firstHuffCode[i] = t_firstCodeDiff[i] = -1;
			}
		}

		t_limit[t_maxHuffLen] = 1ull << 32;
		lastHuffCode = t_firstHuffCode[t_maxHuffLen];
		lastHuffCodeLen = t_maxHuffLen;
		for (int32_t i = t_maxHuffLen - 1; i >= t_minHuffLen; i--)
		{
			if (t_cntCodesPerLen[i] > 0)
			{
				t_limit[i] = (((uint64_t)lastHuffCode) << (32 - lastHuffCodeLen));
				lastHuffCode = t_firstHuffCode[i];
				lastHuffCodeLen = i;
			}
			else
			{
				t_limit[i] = (((uint64_t)lastHuffCode) << (32 - lastHuffCodeLen));
			}
		}

		for (int32_t firstBits = 0; firstBits < LOOKUP_SIZE; firstBits++)
		{
			int32_t l = t_minHuffLen;
			uint32_t blockCode = (firstBits << (32 - LOOKUP_BITS));

			while (blockCode >= t_limit[l])
			{
				l++;
			}

			t_first[firstBits] = l;
		}
	}

	void huffDecode(uint32_t* codes, uint32_t cntCodes, uint32_t dictSize, uint32_t* stream,
		uint32_t streamLen)
	{
		uint32_t tmpCodes[CODES_BLOCK_SIZE];
		uint32_t tmpLens[CODES_BLOCK_SIZE];

		uint64_t* stream64 = (uint64_t*)stream;

		uint64_t bitStream = stream64[0];
		int32_t bitStreamRequiredShift = 0, streamPos = 0;
		int32_t lookupShift = 64 - LOOKUP_BITS;

		__m256i v32 = _mm256_set1_epi32(32);

		for (int32_t codeI = 0; codeI < cntCodes; codeI += CODES_BLOCK_SIZE)
		{
			int32_t codesInBlock = min(CODES_BLOCK_SIZE, cntCodes - codeI);

			for (int32_t i = 0; i < codesInBlock; i += CODES_PER_BITSTREAM)
			{
				for (int32_t j = 0; j < CODES_PER_BITSTREAM; j++)
				{
					uint32_t blockCode = bitStream >> 32;
					int32_t l = t_first[bitStream >> lookupShift];

					while (blockCode >= t_limit[l])
					{
						l++;
					}

					uint32_t curCodeI = j + i;

					bitStreamRequiredShift += l;
					bitStream <<= l;

					tmpCodes[curCodeI] = blockCode;
					tmpLens[curCodeI] = l;
				}

				if (bitStreamRequiredShift > 64)
				{
					bitStreamRequiredShift -= 64;
					streamPos++;
					bitStream |= stream64[streamPos] << bitStreamRequiredShift;
				}
				bitStream |= stream64[streamPos + 1] >> (64 - bitStreamRequiredShift);
			}

			for (int32_t i = 0; i < CODES_BLOCK_SIZE; i += 8)
			{
				// load intermediate value of `blockCode` and `l`
				__m256i codes_8 = _mm256_loadu_si256((__m256i*) & tmpCodes[i]);
				__m256i lens_8 = _mm256_loadu_si256((__m256i*) & tmpLens[i]);

				// blockCode >>= 32 - l
				__m256i shifted_8 = _mm256_srlv_epi32(codes_8, _mm256_sub_epi32(v32, lens_8));
				// load (lookup) value of t_firstCodeDiff[l]
				__m256i lookup_8 = _mm256_i32gather_epi32((const int32_t*)t_firstCodeDiff, lens_8, 4);

				// blockCode += t_firstCodeDiff[l]
				__m256i result_8 = _mm256_add_epi32(shifted_8, lookup_8);
				// store decoded codes
				_mm256_storeu_si256((__m256i*) & codes[codeI + i], result_8);
			}
		}
	}
}
