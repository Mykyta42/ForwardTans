#include <iostream>
#include <array>
#include <immintrin.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <windows.h>
#include <psapi.h>
#include <queue>
#include <iomanip>
#include <filesystem>
#include <cstdlib>

using namespace std;

size_t getMemoryUsage()
{
    PROCESS_MEMORY_COUNTERS info;

    GetProcessMemoryInfo(
        GetCurrentProcess(),
        &info,
        sizeof(info)
    );

    return info.WorkingSetSize;
}

// --- Fast large Huffman integration (adapted from Opt_1_3_4.cpp) ---
namespace Huff
{
#define HUFF_TABLE_CAPACITY (1 << 10)
#define ARRAY_BUFF (256 + 5)

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

    void huffEncode(const vector<uint32_t>& codes, const vector<uint32_t>& frequencies, uint32_t cntCodes, uint32_t dictSize, uint32_t* stream,
        uint32_t& streamLen, uint8_t* huffTable, uint32_t& huffTableLen)
    {
        vector <array<int32_t, 2> > cntPar(2 * dictSize - 1); // weight, parent
        for (auto& it : cntPar) { it[0] = 0; it[1] = -1; }

        // count frequencies
        for (int32_t i = 0; i < dictSize; i++)
        {
			cntPar[i][0] = frequencies[i];
        }

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


#define CODES_PER_BITSTREAM 3
#define CODES_BLOCK_SIZE (CODES_PER_BITSTREAM * 64u)

#define LOOKUP_BITS 12
#define LOOKUP_SIZE (1 << LOOKUP_BITS)
#define LOOKUP_MASK (LOOKUP_SIZE - 1)


    uint32_t t_minHuffLen, t_maxHuffLen;
    uint32_t t_cntCodesPerLen[32];
    uint32_t t_firstCode[32], t_firstHuffCode[32];
    uint32_t t_firstCodeDiff[32]; // = t_firstCode[i] - t_firstHuffCode[i] (2 operation -> 1 operation)
    uint64_t t_limit[32];

    int8_t t_first[LOOKUP_SIZE];

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

    void huffDecode(vector<uint32_t>& codes, uint32_t cntCodes, uint32_t dictSize, uint32_t* stream,
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

// --- end FastHuff ---

struct IndexedTokens {
    vector<uint32_t> ids;
    vector<string> dict;
    vector<uint32_t> fs;
};

IndexedTokens buildIndexLast(const vector<string>& tokens)
{
    int n = tokens.size();

    unordered_map<string, int> map;
    map.reserve(n);

    vector<string> dict;
    vector<uint32_t> ids(n);
    vector<uint32_t> fs;

    int next = 0;

    for (int i = n - 1; i >= 0; i--)
    {
        const string& t = tokens[i];

        auto it = map.find(t);

        if (it == map.end())
        {
            map[t] = next;
            dict.push_back(t);
            ids[i] = next;
            fs.push_back(1);
            next++;
        }
        else
        {
            ids[i] = it->second;
            fs[ids[i]]++;
        }
    }
    return { ids, dict, fs };
}

IndexedTokens buildIndexFreq(const vector<string>& tokens)
{
    unordered_map<string, int> freqMap;

    for (const auto& t : tokens)
        freqMap[t]++;

    vector<pair<string, int>> items;
    items.reserve(freqMap.size());

    for (const auto& p : freqMap)
        items.push_back(p);

    sort(items.begin(), items.end(),
        [](const auto& a, const auto& b)
        {
            if (a.second != b.second)
                return a.second > b.second;
            return a.first < b.first;
        });

    vector<string> dict;
    vector<uint32_t> fs;
    dict.reserve(items.size());
    fs.reserve(items.size());

    unordered_map<string, uint32_t> idMap;
    idMap.reserve(items.size());

    for (uint32_t i = 0; i < items.size(); ++i)
    {
        dict.push_back(items[i].first);
        fs.push_back(items[i].second);
        idMap[items[i].first] = i;
    }

    vector<uint32_t> ids(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i)
        ids[i] = idMap[tokens[i]];

    return { ids, dict, fs };
}

IndexedTokens buildIndexSort(const vector<string>& tokens)
{
    unordered_map<string, int> freqMap;
    vector<string> dict;
    vector<uint32_t> fs;
    vector<uint32_t> ids(tokens.size());
    for (const auto& t : tokens)
        freqMap[t]++;
    dict.reserve(freqMap.size());
    fs.reserve(freqMap.size());
    for (const auto& [token, _] : freqMap)
        dict.push_back(token);
    sort(dict.begin(), dict.end());
    unordered_map<string, int> idMap;
    idMap.reserve(freqMap.size());
    for (int i = 0; i < dict.size(); i++)
        idMap[dict[i]] = i;
    for (int i = 0; i < tokens.size(); i++)
        ids[i] = idMap[tokens[i]];
    for (int i = 0; i < dict.size(); i++)
        fs.push_back(freqMap[dict[i]]);
    return { ids, dict, fs };
}

struct EncodedData {
    vector<string> dictionary;
    vector<uint32_t> freq;
    uint64_t bitCount;
};

size_t compressedSize(const EncodedData& data)
{
    return data.bitCount / 8 + (data.bitCount % 8 != 0);
}

size_t Enthropy(const IndexedTokens& indexed)
{
    size_t total = indexed.ids.size();
    size_t f = indexed.fs.size();
    double entrophy = 0.0;
    for (int i = 0; i < f; i++)
        entrophy += indexed.fs[i] * log2(static_cast<double>(total) / indexed.fs[i]);
    return static_cast<size_t>(entrophy) / 8;
}

enum TokenizationMode {
    CHARWISE = 1,
    SPACES = 2,
    WORDS = 3
};

enum Algorithm {
    UNIFORM_TANS = 1,
    RANGED_TANS = 2,
    FORWARD_TANS = 3,
    ARITHMETIC_FORWARD = 4,
    ARITHMETIC_BACKWARD = 5,
    HUFFMAN_CANONICAL = 6,
    STATIC_ARITHMETIC = 7
};
uint64_t final_state;
uint64_t input_size;

string readFile(const string& name) {
    ifstream f(name, ios::binary);
    if (!f)
        return string();

    f.seekg(0, ios::end);
    auto size = f.tellg();
    if (size <= 0)
        return string();

    f.seekg(0, ios::beg);
    string result;
    result.resize(static_cast<size_t>(size));
    f.read(&result[0], size);
    return result;
}



void writeFile(const string& name, const vector<uint32_t>& data, const vector<string>& W) {

    ofstream f(name, ios::binary);
    if (!f) return;
    if (!data.empty()) {
        size_t totalLen = 0;
        for (int i = 0; i != data.size(); i++) totalLen += W[data[i]].size();
        string F;
        F.reserve(totalLen);
        for (int i = 0; i != data.size(); i++) F.append(W[data[i]]);
        f.write(F.data(), totalLen);
    }

}

string makeCompressedName(const string& input) {
    return input + ".arc";
}

string makeDecompressedName(const string& input) {

    if (input.size() > 4 && input.substr(input.size() - 4) == ".arc")
        return input.substr(0, input.size() - 4) + ".decoded";

    return input + ".decoded";
}

vector<string> tokenize_charwise(const string& text) {

    vector<string> tokens;

    tokens.reserve(text.size());
    for (char c : text)
        tokens.emplace_back(1, c);

    return tokens;
}

vector<string> tokenize_space(const string& text)
{
    vector<string> tokens;

    tokens.reserve(text.size() / 2 + 1);

    size_t i = 0;
    size_t n = text.size();

    while (i < n)
    {
        size_t start = i;

        if (isspace((unsigned char)text[i]))
        {
            while (i < n && isspace((unsigned char)text[i]))
                i++;
        }
        else
        {
            while (i < n && !isspace((unsigned char)text[i]))
                i++;
        }

        tokens.push_back(text.substr(start, i - start));
    }

    return tokens;
}

vector<string> tokenize_words(const string& text)
{
    vector<string> tokens;

    tokens.reserve(text.size() / 2 + 1);

    size_t i = 0;
    size_t n = text.size();
	bool inWord = false;

    while (i < n)
    {
        size_t start = i;

        if (isalnum((unsigned char)text[i]))
        {   
			inWord = true;
            while (i < n && isalnum((unsigned char)text[i]))
                i++;
        }
        else if (isspace((unsigned char)text[i]))
        {
			inWord = false;
            while (i < n && isspace((unsigned char)text[i]))
                i++;
        }
        else
        {
			inWord = false;
            i++;
        }
        if (inWord)
        tokens.push_back(text.substr(start, i - start));
    }

    return tokens;
}

vector<string> tokenize(const string& text, int mode) {

    if (mode == CHARWISE) return tokenize_charwise(text);
    if (mode == SPACES) return tokenize_space(text);
    return tokenize_words(text);
}

static inline uint64_t low_mask(int k)
{
    return k == 64 ? ~0ULL : ((1ULL << k) - 1);
}

static inline int clz64(uint64_t x)
{
    unsigned long r;
    _BitScanReverse64(&r, x);
    return r;
}

struct BitBuffer {

    ofstream& out;

    uint8_t buf = 0;
    int filled = 0;
    uint64_t totalBits = 0;

    BitBuffer(ofstream& o) : out(o) {}

    void flush() {
        out.write((char*)&buf, 1);
        buf = 0;
    }

    void put_bits_lsb(uint64_t bits, int k) {

        totalBits += k;

        while (k > 0) {
            int take = min(8 - filled, k);

            buf |= (bits & ((1ULL << take) - 1)) << filled;

            bits >>= take;
            filled += take;
            k -= take;

            if (filled == 8) {
                out.write((char*)&buf, 1);
                buf = 0;
                filled = 0;
            }
        }
    }

    void put_bits_msb(uint64_t bits, int k) {
        totalBits += k;
        while (k > 0) {
            int take = min(8 - filled, k);
            buf <<= take;
            buf |= (bits >> (k - take)) & ((1ULL << take) - 1);
            filled += take;
            k -= take;
            if (filled == 8) {
                out.write((char*)&buf, 1);
                buf = 0;
                filled = 0;
            }
        }
    }

    void finish() {

        if (filled > 0)
            out.write((char*)&buf, 1);
    }
};

struct BitReader
{
    vector<uint8_t> data;
    uint64_t bitPos = 0;
    uint64_t bitCount;

    BitReader(const string& file, uint64_t bits)
    {
        ifstream in(file, ios::binary);
        in.seekg(sizeof(uint64_t), ios::beg);
        data.assign(istreambuf_iterator<char>(in), {});
        bitCount = bits;
    }


    inline int readBitmsb()
    {
        if (bitPos >= bitCount)
            return 0;

        uint32_t bit =
            (data[bitPos >> 3] >> (7 - (bitPos & 7))) & 1;

        bitPos++;
        return bit;
    }

    inline int getBitlsb(int pos) const
    {
        return (data[pos >> 3] >> (pos & 7)) & 1;
    }

    inline uint64_t peekBits(int k)
    {
        uint64_t result = 0;
        int available = min((uint64_t)k, bitCount);
        for (int i = 0; i < available; i++)
        {
            result <<= 1;
            result |= getBitlsb(bitCount - 1);
            bitCount--;
        }

        return result;
    }
};



static inline void renorm_and_put(uint64_t& x,
    uint64_t threshold,
    BitBuffer& bw)
{
    int k = max(0, clz64(x) - clz64(threshold));

    if ((x >> k) >= threshold)
        ++k;

    uint64_t out = x & low_mask(k);

    x >>= k;

    bw.put_bits_lsb(out, k);
}

static inline void renorm_and_get(uint64_t& x,
    uint64_t l,
    BitReader& br)
{
    int k = max(0, clz64(l) - clz64(x));
    int old = br.bitCount;

    uint64_t bits = br.peekBits(k);

    x = (x << (old - br.bitCount)) | bits;

    if (x < l)
    {
        x = (x << (br.bitCount > 0)) | br.peekBits(1);
    }
}

struct Fenwick {
    int n;
    uint64_t total;
    int power2;
    vector<uint64_t> t;

    Fenwick(int n) : n(n), t(n + 1, 0), total(0), power2(1) {
        while ((power2 << 1) <= n) power2 <<= 1;
    }

    void add(int i, int delta) {
        for (++i; i <= n; i += i & -i)
            t[i] += delta;
        total += delta;
    }

    uint64_t sum(int i) {
        uint64_t s = 0;
        for (++i; i > 0; i -= i & -i)
            s += t[i];
        return s;
    }

    int find(uint32_t x)
    {

        int idx = 0;

        for (int bit = power2; bit; bit >>= 1)
        {
            int next = idx + bit;

            if (next <= n && t[next] <= x)
            {
                x -= t[next];
                idx = next;
            }
        }

		return idx;
    }
};

EncodedData ArithmeticBackward(const IndexedTokens& data,
    const string& outFile)
{
    const vector<uint32_t>& T = data.ids;
    int sigma = data.dict.size();

    Fenwick fw(sigma);

    for (int i = 0; i < sigma; i++)
        fw.add(i, 1);

    ofstream out(outFile, ios::binary);
    BitBuffer bw(out);
    uint64_t bitCount = 0;
    out.write((char*)&bitCount, sizeof(uint64_t));

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t pending = 0;

    auto output_bit = [&](int b) {
        bw.put_bits_msb(b, 1);
        while (pending--) bw.put_bits_msb(!b, 1);
        pending = 0;
        };

    for (int s : T) {
        uint64_t total = fw.total;

        uint64_t symLow = (s == 0 ? 0 : fw.sum(s - 1));
        uint64_t symHigh = fw.sum(s);

        uint64_t range = high - low + 1;

        high = low + (range * symHigh) / total - 1;
        low = low + (range * symLow) / total;

        while (true) {
            if (high < HALF) {
                output_bit(0);
            }
            else if (low >= HALF) {
                output_bit(1);
                low -= HALF; high -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                pending++;
                low -= QUARTER; high -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
        }

        fw.add(s, +1);
    }

    pending++;
    output_bit(low < QUARTER ? 0 : 1);

    for (int i = 0; i < 32; ++i)
        bw.put_bits_msb(0, 1);

    bw.finish();
    bitCount = bw.totalBits;
    input_size = T.size();
    out.seekp(0);
    out.write((char*)&bitCount, sizeof(uint64_t));

    return { data.dict, data.fs, bw.totalBits };
}

EncodedData ArithmeticForward(const IndexedTokens& data,
    const string& outFile)
{
    const vector<uint32_t>& T = data.ids;
    int sigma = data.fs.size();
    vector<int> redfreqs(sigma + 1);
    for (int i = 0; i < sigma; i++) {
        redfreqs[i] = data.fs[i] - 1;
    }
    redfreqs[sigma] = sigma;

    Fenwick fw(sigma + 1);

    for (int i = 0; i < sigma + 1; i++)
        fw.add(i, redfreqs[i]);
    ofstream out(outFile, ios::binary);
    BitBuffer bw(out);
    uint64_t bitCount = 0;
    out.write((char*)&bitCount, sizeof(uint64_t));

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t pending = 0;

    auto output_bit = [&](int b) {
        bw.put_bits_msb(b, 1);
        while (pending--) bw.put_bits_msb(!b, 1);
        pending = 0;
        };

    for (int s : T) {
        uint64_t total = fw.total;
        int s1 = redfreqs[s]-- > 0 ? s : sigma;
        uint64_t symLow = (s1 == 0 ? 0 : fw.sum(s1 - 1));
        uint64_t symHigh = fw.sum(s1);
        uint64_t range = high - low + 1;

        high = low + (range * symHigh) / total - 1;
        low = low + (range * symLow) / total;

        while (true) {
            if (high < HALF) {
                output_bit(0);
            }
            else if (low >= HALF) {
                output_bit(1);
                low -= HALF; high -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                pending++;
                low -= QUARTER; high -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
        }

        fw.add(s1, -1);

    }
    pending++;
    output_bit(low < QUARTER ? 0 : 1);

    for (int i = 0; i < 32; ++i)
        bw.put_bits_msb(0, 1);

    bw.finish();
    bitCount = bw.totalBits;
    input_size = T.size();
    out.seekp(0);
    out.write((char*)&bitCount, sizeof(uint64_t));

    return { data.dict, data.fs, bw.totalBits };
}

EncodedData ArithmeticStatic(const IndexedTokens& data, const string& outFile) {
    const vector<uint32_t>& T = data.ids;
    int sigma = data.fs.size();

    Fenwick fw(sigma);

    for (int i = 0; i < sigma; i++)
        fw.add(i, data.fs[i]);
    ofstream out(outFile, ios::binary);
    BitBuffer bw(out);
    uint64_t bitCount = 0;
    out.write((char*)&bitCount, sizeof(uint64_t));

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t pending = 0;

    auto output_bit = [&](int b) {
        bw.put_bits_msb(b, 1);
        while (pending--) bw.put_bits_msb(!b, 1);
        pending = 0;
        };

    for (int s : T) {
        uint64_t total = fw.total;
        uint64_t symLow = (s == 0 ? 0 : fw.sum(s - 1));
        uint64_t symHigh = fw.sum(s);
        uint64_t range = high - low + 1;

        high = low + (range * symHigh) / total - 1;
        low = low + (range * symLow) / total;

        while (true) {
            if (high < HALF) {
                output_bit(0);
            }
            else if (low >= HALF) {
                output_bit(1);
                low -= HALF; high -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                pending++;
                low -= QUARTER; high -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
        }

    }
    pending++;
    output_bit(low < QUARTER ? 0 : 1);

    for (int i = 0; i < 32; ++i)
        bw.put_bits_msb(0, 1);

    bw.finish();
    bitCount = bw.totalBits;
    out.seekp(0);
    input_size = T.size();
    out.write((char*)&bitCount, sizeof(uint64_t));

    return { data.dict, data.fs, bw.totalBits };
}


EncodedData HuffmanCanonical(const IndexedTokens& data, const string& baseName)
{
	size_t g_cntWords = data.ids.size();

		uint32_t streamLen = 0;
		uint32_t* stream = new uint32_t[g_cntWords];

		uint32_t huffTableLen = 0;
		uint8_t* huffTable = new uint8_t[HUFF_TABLE_CAPACITY];

		Huff::huffEncode(data.ids, data.fs, g_cntWords, data.dict.size(), stream, streamLen, huffTable, huffTableLen);

		ofstream fEncoded(baseName, ios::binary);
		fEncoded.write((char*)stream, streamLen * sizeof(uint32_t));
		fEncoded.close();

		delete[] stream;

		string huffTableFileName = baseName + ".tbl";
		ofstream fHuffTable(huffTableFileName, ios::binary);
		fHuffTable.write((char*)huffTable, huffTableLen);
		fHuffTable.close();
		delete[] huffTable;
        input_size = g_cntWords;

    return { data.dict, data.fs, streamLen * 32 };
}

EncodedData UniformTans(const IndexedTokens& T, const string& outFile) {

    uint64_t n = T.ids.size();
    uint64_t f = T.fs.size();
    uint64_t cumulativeFreq = 0;
    vector<uint64_t> c(f);
    vector<uint64_t> fn(f);
    int L;

    auto cmp = [&](const pair<uint64_t, uint64_t>& left, const pair<uint64_t, uint64_t>& right) {
        uint64_t lden = fn[left.second];
        uint64_t rden = fn[right.second];
		return static_cast<float>(2 * left.first + 1) / (2 * lden) > static_cast<float>(2 * right.first + 1) / (2 * rden); // Duda's version with float division to avoid overflow
        };

    priority_queue<pair<uint64_t, uint64_t>, vector<pair<uint64_t, uint64_t>>, decltype(cmp)> q(cmp);

	if (f <= 256) {
		vector<int> idx(f);
		L = 1 << 12;
        int norm_n = 0;
        for (int i = 0; i < f; ++i) {
            fn[i] = max(1, T.fs[i] * L / n);
            idx[i] = i;
            norm_n += fn[i];
        }

        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fn[a] > fn[b];
            });

        int i = 0;
        while (norm_n < L) {
            fn[idx[i]]++;
            norm_n++;
            i++;
            if (i == f) i = 0;
        }
        while (norm_n > L) {
            if (fn[idx[i]] > 1) fn[idx[i]]--;
            norm_n--;
            i++;
            if (i == f) i = 0;
        }

        for (size_t i = 0; i < f; ++i) {
            c[i] = cumulativeFreq;
            cumulativeFreq += fn[i];
        }
	}

    else {
        L = n;
        for (size_t i = 0; i < f; ++i) {
            c[i] = cumulativeFreq;
            cumulativeFreq += T.fs[i];
			fn[i] = T.fs[i];
        }
        
    }
    vector<uint64_t> I(L);

    for (uint64_t i = 0; i < f; ++i)
        q.push({ 0, i });

    for (uint64_t i = 0; i < L; i++) {
        auto p = q.top(); q.pop();
        I[c[p.second] + p.first] = i;
        q.push({ p.first + 1, p.second });
    }
    ofstream out(outFile, ios::binary);
    uint64_t bitCount = 0;
    out.write((char*)&bitCount, sizeof(uint64_t));
    uint64_t x = L;
    BitBuffer bw(out);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t idx = T.ids[i];
        renorm_and_put(x, 2 * fn[idx], bw);
        x = L + I[c[idx] + x - fn[idx]];
    }
    bw.finish();
    bitCount = bw.totalBits;
    out.seekp(0);
    out.write((char*)&bitCount, sizeof(uint64_t));
    final_state = x;
    input_size = n;
    return { T.dict, T.fs, bw.totalBits };
}




EncodedData ForwardTans(const IndexedTokens& T, const string& outFile) {

    uint64_t n = T.ids.size();
    uint64_t w = T.fs.size();
    uint64_t cumulativeFreq = 0;
    vector<uint64_t> I(n + w);
    vector<uint64_t> c(w + 1, 0);
    vector<uint64_t> pos(w + 1, 0);

    for (size_t i = 1; i <= w; ++i) {
        cumulativeFreq += T.fs[i - 1];
        c[i] = cumulativeFreq;
    }

    for (uint64_t i = n; i > 0; i--) {
        int idx = T.ids[i - 1];
        if (!pos[idx]) {
            I[n + pos[w]] = n - i + pos[w];
            pos[w]++;
        }
        I[c[idx] + pos[idx]] = n - i + pos[w];
        if (pos[idx] + 1 != T.fs[idx]) pos[idx]++;
    }

    uint64_t l = n + w;
    ofstream out(outFile, ios::binary);
    uint64_t bitCount = 0;
    out.write((char*)&bitCount, sizeof(uint64_t));
    uint64_t x = l;
    BitBuffer bw(out);

    for (uint64_t i = 0; i < n; i++) {
        uint64_t idx = T.ids[i];
        if (!pos[idx]) {
            l--;
            idx = w;
        }
        renorm_and_put(x, 2 * pos[idx], bw);
        x = l + I[c[idx] + x - pos[idx]];
        pos[idx]--;
        l--;
    }

    bw.finish();
    bitCount = bw.totalBits;
    final_state = x;
	input_size = n;
    out.seekp(0);
    out.write((char*)&bitCount, sizeof(uint64_t));
    return { T.dict, T.fs, bw.totalBits };
}
EncodedData RangedTans(const IndexedTokens& T,
    const string& outFile)
{
    const uint64_t n = T.ids.size();
    const uint64_t sigma = T.fs.size();

    if (n == 0)
        return { T.dict, T.fs, 0 };


    vector<uint64_t> c(sigma);
    uint64_t total = 0;

    for (uint64_t a = 0; a < sigma; ++a) {
        c[a] = total;
        total += T.fs[a];
    }

    ofstream out(outFile, ios::binary);


    uint64_t bitCount = 0;
    out.write(
        reinterpret_cast<const char*>(&bitCount),
        sizeof(bitCount));

    uint64_t x = n;
    uint64_t byteCount = 0;

    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t a = T.ids.at(i);


        const uint64_t freq = T.fs.at(a);
        const uint64_t threshold = 256ULL * freq;

        while (x >= threshold) {
            const uint8_t byte =
                static_cast<uint8_t>(x & 0xFFULL);

            out.write(
                reinterpret_cast<const char*>(&byte), 1);

            ++byteCount;
            x >>= 8;
        }

        const uint64_t quotient = x / freq;
        const uint64_t remainder = x % freq;

        x = n * quotient + c[a] + remainder;

    }

    bitCount = 8 * byteCount;
    final_state = x;
    input_size = n;

    out.seekp(0, ios::beg);
    out.write(
        reinterpret_cast<const char*>(&bitCount),
        sizeof(bitCount));


    out.close();

    return { T.dict, T.fs, bitCount };
}

EncodedData encode(const IndexedTokens& indexed, int algorithm, const string& codeFile)
{
    if (algorithm == UNIFORM_TANS) return UniformTans(indexed, codeFile);
    if (algorithm == RANGED_TANS) return RangedTans(indexed, codeFile);
    if (algorithm == ARITHMETIC_FORWARD) return ArithmeticForward(indexed, codeFile);
    if (algorithm == ARITHMETIC_BACKWARD) return ArithmeticBackward(indexed, codeFile);
    if (algorithm == HUFFMAN_CANONICAL) return HuffmanCanonical(indexed, codeFile);
    if (algorithm == STATIC_ARITHMETIC) return ArithmeticStatic(indexed, codeFile);
    return ForwardTans(indexed, codeFile);
}

vector<uint32_t> UANSdecode(const EncodedData& data, const string& baseName) {
    uint64_t n = input_size;
    uint64_t f = data.freq.size();
    uint64_t cumulativeFreq = 0;
    vector<uint64_t> c(f);
	vector<uint64_t> fn(f);
    int L;
    vector <uint32_t> tokensids(n);

    auto cmp = [&](const pair<uint64_t, uint64_t>& left, const pair<uint64_t, uint64_t>& right) {
        uint64_t lden = fn[left.second];
        uint64_t rden = fn[right.second];
        return static_cast<float>(2 * left.first + 1) / (2 * lden) > static_cast<float>(2 * right.first + 1) / (2 * rden);
        };

    priority_queue<pair<uint64_t, uint64_t>, vector<pair<uint64_t, uint64_t>>, decltype(cmp)> q(cmp);

    if (f <= 256) {
        vector<int> idx(f);
        L = 1 << 12;
        int norm_n = 0;
        for (int i = 0; i < f; ++i) {
            fn[i] = max(1, data.freq[i] * L / n);
            idx[i] = i;
            norm_n += fn[i];
        }

        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fn[a] > fn[b];
            });

        int i = 0;
        while (norm_n < L) {
            fn[idx[i]]++;
            norm_n++;
            i++;
            if (i == f) i = 0;
        }
        while (norm_n > L) {
            if (fn[idx[i]] > 1) fn[idx[i]]--;
            norm_n--;
            i++;
            if (i == f) i = 0;
        }

        for (size_t i = 0; i < f; ++i) {
            c[i] = cumulativeFreq;
            cumulativeFreq += fn[i];
        }
    }

    else {
		L = n;
        for (size_t i = 0; i < f; ++i) {
            c[i] = cumulativeFreq;
            cumulativeFreq += data.freq[i];
            fn[i] = data.freq[i];
        }

    }

    vector<pair<uint64_t, uint64_t>> I(L);

    for (uint64_t i = 0; i < f; ++i)
        q.push({ 0, i });

    for (uint64_t i = 0; i < L; i++) {
        auto p = q.top(); q.pop();
        I[i] = { p.first, p.second };
        q.push({ p.first + 1, p.second });
    }
    BitReader br(baseName + ".code", data.bitCount);
    uint64_t x = final_state;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t tokenIndex = I[x - L].second;
        x = I[x - L].first + fn[tokenIndex];
        tokensids[n - 1 - i] = tokenIndex;
        renorm_and_get(x, L, br);
    }
    return tokensids;
}


vector<uint32_t> RANSdecode(const EncodedData& data, const string& baseName) {

    const uint64_t n = input_size;
    const uint64_t sigma = data.freq.size();
    constexpr uint64_t B = 256;
    if (n == 0)
        return {};

    vector<uint64_t> c(sigma);
    vector<uint32_t> symbolOf(n);

    uint64_t cumulativeFreq = 0;

    for (uint64_t a = 0; a < sigma; ++a) {
        c[a] = cumulativeFreq;

        for (uint64_t j = 0; j < data.freq[a]; ++j) {
            symbolOf[cumulativeFreq + j] =
                static_cast<uint32_t>(a);
        }

        cumulativeFreq += data.freq[a];
    }

    const uint64_t L = n;

    BitReader br(baseName + ".code", data.bitCount);
    uint64_t x = final_state;

    vector<uint32_t> result(n);
    
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t slot = x % n;
        const uint32_t a = symbolOf[slot];

        result[n - 1 - i] = a;

        x = static_cast<uint64_t>(data.freq[a]) * (x / n)
            + slot - c[a];

        while (x < L) {
            x = (x << 8) | br.peekBits(8);
        }

    }

    return result;
}

vector<uint32_t> FANSdecode(const EncodedData& data, const string& baseName) {
    BitReader br(baseName + ".code", data.bitCount);
    const vector<string>& W = data.dictionary;
	int n = input_size;
    uint64_t x = 1;
    uint64_t l = 1;
    uint64_t offset = 0;

    vector<pair<uint64_t, uint64_t>> P(n+W.size());
    vector<uint64_t> f(W.size(), 0);
    vector<uint32_t> tokensids(n);
    for (int i = 0; i < n; i++) {
        if (x - l == l - 1 || P[x - l].first == 0) {
            P[i + offset] = { 0, offset };
            P[i + offset + 1] = { offset + 1, 0 };

            f[offset] = 1;
            tokensids[n - 1 - i] = offset;
            offset++;

            x = offset + P[x - l].second;
            l += 2;
            renorm_and_get(x, l, br);
        }
        else {
            uint64_t tokenIndex = P[x - l].first - 1;

            P[i + offset] = { P[x - l].first, f[tokenIndex] };

            tokensids[n - 1 - i] = tokenIndex;
            x = f[tokenIndex] + P[x - l].second;
            f[tokenIndex]++;
            l++;
            renorm_and_get(x, l, br);
        }
    }
    return tokensids;
}



void writeCompressed(const string& baseName,
    const EncodedData& data,
    int mode)
{
    {
        ofstream out(baseName + ".dict", ios::binary);

        int dictSize = (int)data.dictionary.size();

        out.write((char*)&dictSize, sizeof(int));
        out.write((char*)&mode, sizeof(int));

        bool used[256] = {};

        for (const auto& w : data.dictionary)
            for (unsigned char c : w)
                used[c] = true;

        unsigned char delim = 0;
        while (delim < 255 && used[delim])
            ++delim;

        out.write((char*)&delim, 1);

        for (const auto& w : data.dictionary)
        {
            out.write(w.data(), w.size());
            out.put((char)delim);
        }
    }

    if (!data.freq.empty())
    {
        ofstream out(baseName + ".freq", ios::binary);

        int size = (int)data.freq.size();
        out.write((char*)&size, sizeof(int));

        for (uint32_t f : data.freq)
            out.write((char*)&f, sizeof(uint32_t));
    }
}

EncodedData readCompressed(const string& baseName, int& mode)
{
    vector<string> dict;
    vector<uint32_t> freq;
    uint64_t bitCount;

    {
        ifstream in(baseName + ".dict", ios::binary);

        int dictSize;

        in.read((char*)&dictSize, sizeof(int));
        in.read((char*)&mode, sizeof(int));

        unsigned char delim;
        in.read((char*)&delim, 1);

        dict.reserve(dictSize);

        string cur;
        char ch;

        while ((int)dict.size() < dictSize && in.get(ch))
        {
            if ((unsigned char)ch == delim)
            {
                dict.push_back(std::move(cur));
                cur.clear();
            }
            else
            {
                cur.push_back(ch);
            }
        }
    }

    {
        ifstream in(baseName + ".code", ios::binary);
        in.read((char*)&bitCount, sizeof(uint64_t));
    }

    {
        ifstream in(baseName + ".freq", ios::binary);

        if (in)
        {
            int size;
            in.read((char*)&size, sizeof(int));

            freq.resize(size);
            for (int i = 0; i < size; i++)
                in.read((char*)&freq[i], sizeof(uint32_t));
        }
    }

    return { dict, freq, bitCount };
}

vector<uint32_t> FACdecode(const EncodedData& data, const string& baseName)
{
    BitReader br(baseName + ".code", data.bitCount);
    const int sigma = data.freq.size();
    int n = input_size;
    vector<uint32_t> redfreqs(sigma + 1);
    for (int i = 0; i < sigma; i++)
        redfreqs[i] = data.freq[i] - 1;

    redfreqs[sigma] = sigma;
    int nyt = sigma - 1;

    Fenwick fw(sigma + 1);

    for (int i = 0; i < sigma + 1; i++)
        fw.add(i, redfreqs[i]);

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t code = 0;

    for (int i = 0; i < 32; i++) {
        int b = br.readBitmsb();
        code = (code << 1) | b;
    }
    auto renorm = [&]() {
        while (true) {
            if (high < HALF) {
            }
            else if (low >= HALF) {
                low -= HALF;
                high -= HALF;
                code -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                low -= QUARTER;
                high -= QUARTER;
                code -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | br.readBitmsb();
        }
        };

    vector<uint32_t> result(n);

    for (int i = 0; i < n; i++)
    {
        uint64_t total = fw.total;

        uint64_t range = high - low + 1;

        uint64_t scaled =
            ((code - low + 1) * total - 1) / range;

        int s1 = fw.find(scaled);

        int s;

        if (s1 == sigma)
        {
            s = nyt--;
        }
        else
        {
            s = s1;
        }

        result[i] = s;

        uint64_t lowCum = fw.sum(s1 - 1);
        uint64_t highCum = fw.sum(s1);
        fw.add(s1, -1);


        high = low + (range * highCum) / total - 1;
        low = low + (range * lowCum) / total;

        renorm();
    }

    return result;
}

vector<uint32_t> BACdecode(const EncodedData& data, const string& baseName)
{
    BitReader br(baseName + ".code", data.bitCount);
    const int sigma = data.freq.size();
    uint64_t n = input_size;

    Fenwick fw(sigma);

    for (int i = 0; i < sigma; i++)
        fw.add(i, 1);

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t code = 0;

    for (int i = 0; i < 32; i++) {
        int b = br.readBitmsb();
        code = (code << 1) | b;
    }
    auto renorm = [&]() {
        while (true) {
            if (high < HALF) {
            }
            else if (low >= HALF) {
                low -= HALF;
                high -= HALF;
                code -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                low -= QUARTER;
                high -= QUARTER;
                code -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | br.readBitmsb();
        }
        };

    vector<uint32_t> result(n);

    for (int i = 0; i < n; i++)
    {
        uint64_t total = fw.total;

        uint64_t range = high - low + 1;

        uint64_t scaled =
            ((code - low + 1) * total - 1) / range;

        int s = fw.find(scaled);

        result[i] = s;

        uint64_t lowCum = fw.sum(s - 1);
        uint64_t highCum = fw.sum(s);
        fw.add(s, 1);


        high = low + (range * highCum) / total - 1;
        low = low + (range * lowCum) / total;

        renorm();
    }

    return result;
}

vector<uint32_t> SACdecode(const EncodedData& data, const string& baseName)
{
    BitReader br(baseName + ".code", data.bitCount);
    const int sigma = data.freq.size();

    Fenwick fw(sigma);

    for (int i = 0; i < sigma; i++)
        fw.add(i, data.freq[i]);

    const uint64_t TOP = 0xFFFFFFFF;
    const uint64_t HALF = 1ULL << 31;
    const uint64_t QUARTER = 1ULL << 30;

    uint64_t low = 0, high = TOP;
    uint64_t code = 0;

    for (int i = 0; i < 32; i++) {
        int b = br.readBitmsb();
        code = (code << 1) | b;
    }
    auto renorm = [&]() {
        while (true) {
            if (high < HALF) {
            }
            else if (low >= HALF) {
                low -= HALF;
                high -= HALF;
                code -= HALF;
            }
            else if (low >= QUARTER && high < 3 * QUARTER) {
                low -= QUARTER;
                high -= QUARTER;
                code -= QUARTER;
            }
            else break;

            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | br.readBitmsb();
        }
        };

    int n = fw.total;
    vector<uint32_t> result(n);
    for (int i = 0; i < n; i++)
    {
        uint64_t total = fw.total;

        uint64_t range = high - low + 1;

        uint64_t scaled =
            ((code - low + 1) * total - 1) / range;

        int s = fw.find(scaled);


        result[i] = s;

        uint64_t lowCum = fw.sum(s - 1);
        uint64_t highCum = fw.sum(s);

        high = low + (range * highCum) / total - 1;
        low = low + (range * lowCum) / total;

        renorm();
    }

    return result;
}

vector<uint32_t> HuffmanCanonicaldecode(const EncodedData& data, const string& baseName)
{
	uint64_t g_cntWords = input_size;
	uint32_t g_dictSize = data.freq.size();
    
    vector<uint32_t> codes(g_cntWords + ARRAY_BUFF);

    uint32_t streamLen;
    uint32_t* stream;

    string encodedFileName = baseName + ".code";
    ifstream fEncoded(encodedFileName, ios::binary);
    fEncoded.seekg(0, ios::end);
    streamLen = fEncoded.tellg() / sizeof(uint32_t);
    fEncoded.seekg(0, ios::beg);
    stream = new uint32_t[streamLen];
    fEncoded.read((char*)stream, streamLen * sizeof(uint32_t));
    fEncoded.close();

    uint32_t huffTableLen;
    uint8_t* huffTable;

    string huffTableFileName = baseName + ".code.tbl";
    ifstream fHuffTable(huffTableFileName, ios::binary);
    fHuffTable.seekg(0, ios::end);
    huffTableLen = fHuffTable.tellg();
    fHuffTable.seekg(0, ios::beg);
    huffTable = new uint8_t[huffTableLen];
    fHuffTable.read((char*)huffTable, huffTableLen);
    fHuffTable.close();
    Huff::precompute(huffTable, huffTableLen, g_dictSize);

    delete[] huffTable;
    Huff::huffDecode(codes, g_cntWords, g_dictSize, stream, streamLen);
    delete[] stream;

    return vector<uint32_t>(codes.begin(), codes.begin() + g_cntWords);
}


string algorithmName(int algorithm)
{
    if (algorithm == UNIFORM_TANS) return "Uniform Tans";
    if (algorithm == RANGED_TANS) return "Ranged Tans";
    if (algorithm == FORWARD_TANS) return "Forward Tans";
    if (algorithm == ARITHMETIC_FORWARD) return "Arithmetic Forward";
    if (algorithm == ARITHMETIC_BACKWARD) return "Arithmetic Backward";
    if (algorithm == HUFFMAN_CANONICAL) return "Huffman Canonical";
    if (algorithm == STATIC_ARITHMETIC) return "Static Arithmetic";
    return "Unknown";
}

string tokenizationName(int mode)
{
    if (mode == CHARWISE) return "Character-wise";
    if (mode == SPACES) return "Space-based";
    if (mode == WORDS) return "Word-based";
    return "Other";
}

string removeUtf8Bom(string s)
{
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF)
    {
        s.erase(0, 3);
    }
    return s;
}

string trim(const string& s)
{
    size_t b = 0, e = s.size();
    while (b < e && isspace((unsigned char)s[b])) b++;
    while (e > b && isspace((unsigned char)s[e - 1])) e--;
    string r = s.substr(b, e - b);
    r = removeUtf8Bom(r);
    if (r.size() >= 2 && ((r.front() == '"' && r.back() == '"') || (r.front() == '\'' && r.back() == '\'')))
        r = r.substr(1, r.size() - 2);
    return r;
}

vector<string> readFileListCsv(const string& csvName)
{
    ifstream in(csvName);
    vector<string> files;
    string line;

    while (getline(in, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = removeUtf8Bom(line);

        size_t comma = line.find(',');
        string name = trim(comma == string::npos ? line : line.substr(0, comma));
        if (name.empty()) continue;

        string lower = name;
        transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)tolower(c); });
        if (lower == "file" || lower == "filename" || lower == "name") continue;

        files.push_back(name);
    }
    return files;
}

string xmlEscape(const string& s)
{
    string r;
    r.reserve(s.size());
    for (char ch : s)
    {
        unsigned char c = (unsigned char)ch;
        if (c == '&') r += "&amp;";
        else if (c == '<') r += "&lt;";
        else if (c == '>') r += "&gt;";
        else if (c == '"') r += "&quot;";
        else if (c == '\'') r += "&apos;";
        else if (c >= 32 || c == '\t' || c == '\n' || c == '\r') r += ch;
    }
    return r;
}

string columnName(int col)
{
    string s;
    while (col > 0)
    {
        int rem = (col - 1) % 26;
        s.push_back(char('A' + rem));
        col = (col - 1) / 26;
    }
    reverse(s.begin(), s.end());
    return s;
}

string safeBaseName(string s)
{
    replace(s.begin(), s.end(), '\\', '_');
    replace(s.begin(), s.end(), '/', '_');
    replace(s.begin(), s.end(), ':', '_');
    replace(s.begin(), s.end(), '*', '_');
    replace(s.begin(), s.end(), '?', '_');
    replace(s.begin(), s.end(), '"', '_');
    replace(s.begin(), s.end(), '<', '_');
    replace(s.begin(), s.end(), '>', '_');
    replace(s.begin(), s.end(), '|', '_');
    return s;
}

uint64_t fileSizeOrZero(const string& fileName)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(fileName, ec)) return 0;
    uintmax_t sz = fs::file_size(fileName, ec);
    if (ec) return 0;
    return static_cast<uint64_t>(sz);
}

uint64_t compressWith7zPPMd(const string& fileName)
{
    namespace fs = std::filesystem;

    if (!fs::exists(fileName)) {
        cout << "File not found: " << fileName << "\n";
        return 0;
    }

    string archiveName = fileName + ".ppmd.7z";
    fs::remove(archiveName);

    string cmd =
        "cmd /C \"\"C:\\Progra~1\\7-Zip\\7z.exe\" a -t7z -m0=PPMd -mx=9 -y -bd -bb0 "
        "\"" + archiveName + "\" "
        "\"" + fileName + "\" "
        ">nul 2>&1\"";

    //cout << "7z command: " << cmd << "\n";

    int rc = system(cmd.c_str());

    /*cout << "7z rc = " << rc << "\n";
    cout << "archive exists = " << fs::exists(archiveName) << "\n";*/

    if (rc != 0 || !fs::exists(archiveName))
        return 0;

    return fileSizeOrZero(archiveName);
}

struct BenchmarkRecord
{
    string filename;
    int mode;
    int algorithm;

    uint64_t codeBytes = 0;             // size of the .code part only
    uint64_t dictPpmdBytes = 0;         // size of .dict compressed separately by 7-Zip PPMd
    uint64_t freqPpmdBytes = 0;         // size of .freq compressed separately by 7-Zip PPMd
    uint64_t totalBytes = 0;            // codeBytes + dictPpmdBytes + freqPpmdBytes

    double compressSeconds = -1.0;
    double decompressSeconds = -1.0; // -1 means unsupported or failed
};

bool hasDecoder(int algorithm)
{
    return algorithm == UNIFORM_TANS ||
        algorithm == RANGED_TANS ||
        algorithm == FORWARD_TANS ||
        algorithm == ARITHMETIC_FORWARD ||
        algorithm == STATIC_ARITHMETIC ||
        algorithm == ARITHMETIC_BACKWARD ||
        algorithm == HUFFMAN_CANONICAL;
}

vector<uint32_t> decodeByAlgorithm(const EncodedData& data, const string& baseName, int algorithm)
{
    if (algorithm == UNIFORM_TANS) return UANSdecode(data, baseName);
    if (algorithm == RANGED_TANS) return RANSdecode(data, baseName);
    if (algorithm == FORWARD_TANS) return FANSdecode(data, baseName);
    if (algorithm == ARITHMETIC_FORWARD) return FACdecode(data, baseName);
	if (algorithm == ARITHMETIC_BACKWARD) return BACdecode(data, baseName);
	if (algorithm == STATIC_ARITHMETIC) return SACdecode(data, baseName);
    if (algorithm == HUFFMAN_CANONICAL) return HuffmanCanonicaldecode(data, baseName);
    return {};
}

void writeOneMetricTable(ofstream& out, int& r, const string& title,
    const vector<BenchmarkRecord>& rows, int mode, int metric)
{
    auto cellStr = [&](int rr, int c, const string& v) {
        out << "<c r=\"" << columnName(c) << rr << "\" t=\"inlineStr\"><is><t>"
            << xmlEscape(v) << "</t></is></c>";
        };
    auto cellNum = [&](int rr, int c, double v) {
        out << "<c r=\"" << columnName(c) << rr << "\"><v>"
            << fixed << setprecision(metric <= 4 ? 0 : 6) << v << "</v></c>";
        };

    out << "<row r=\"" << r << "\">"; cellStr(r, 1, title); out << "</row>\n";
    r++;
    out << "<row r=\"" << r << "\">"; cellStr(r, 1, "File");
    for (int a = UNIFORM_TANS; a <= STATIC_ARITHMETIC; a++) cellStr(r, a + 1, algorithmName(a));
    out << "</row>\n";

    vector<string> files;
    for (const auto& x : rows)
        if (x.mode == mode && find(files.begin(), files.end(), x.filename) == files.end())
            files.push_back(x.filename);

    for (const string& f : files)
    {
        r++;
        out << "<row r=\"" << r << "\">";
        cellStr(r, 1, f);
		for (int a = UNIFORM_TANS; a <= STATIC_ARITHMETIC; a++)
        {
            auto it = find_if(rows.begin(), rows.end(), [&](const BenchmarkRecord& x) {
                return x.mode == mode && x.algorithm == a && x.filename == f;
                });
            if (it == rows.end()) continue;
            if (metric == 1 && it->codeBytes > 0) cellNum(r, a + 1, (double)it->codeBytes);
            else if (metric == 2 && it->dictPpmdBytes > 0) cellNum(r, a + 1, (double)it->dictPpmdBytes);
            else if (metric == 3 && it->freqPpmdBytes > 0) cellNum(r, a + 1, (double)it->freqPpmdBytes);
            else if (metric == 4 && it->totalBytes > 0) cellNum(r, a + 1, (double)it->totalBytes);
            else if (metric == 5 && it->compressSeconds >= 0.0) cellNum(r, a + 1, it->compressSeconds);
            else if (metric == 6 && it->decompressSeconds >= 0.0) cellNum(r, a + 1, it->decompressSeconds);
            else cellStr(r, a + 1, "N/A");
        }
        out << "</row>\n";
    }
    r += 3;
}

void writeWorksheetXml(const string& fileName, const vector<BenchmarkRecord>& rows, int mode)
{
    ofstream out(fileName, ios::binary);
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
    out << "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\n<sheetData>\n";
    int r = 1;
    writeOneMetricTable(out, r, tokenizationName(mode) + ": code size, bytes", rows, mode, 1);
    writeOneMetricTable(out, r, tokenizationName(mode) + ": dictionary size after 7-Zip PPMd -mx=9, bytes", rows, mode, 2);
    writeOneMetricTable(out, r, tokenizationName(mode) + ": frequency table size after 7-Zip PPMd -mx=9, bytes", rows, mode, 3);
    writeOneMetricTable(out, r, tokenizationName(mode) + ": total size = code + compressed dictionary + compressed frequencies, bytes", rows, mode, 4);
    writeOneMetricTable(out, r, tokenizationName(mode) + ": compression time, seconds", rows, mode, 5);
    writeOneMetricTable(out, r, tokenizationName(mode) + ": decompression time, seconds", rows, mode, 6);
    out << "</sheetData>\n</worksheet>\n";
}

bool writeXlsxReport(const string& outName, const vector<BenchmarkRecord>& rows)
{
    namespace fs = std::filesystem;
    fs::path temp = fs::temp_directory_path() / ("MyCompressor_xlsx_" + to_string(chrono::high_resolution_clock::now().time_since_epoch().count()));
    fs::create_directories(temp / "_rels");
    fs::create_directories(temp / "xl" / "_rels");
    fs::create_directories(temp / "xl" / "worksheets");

    {
        ofstream out(temp / "[Content_Types].xml", ios::binary);
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            << "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            << "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            << "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            << "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>";
        for (int i = 1; i <= 3; i++)
            out << "<Override PartName=\"/xl/worksheets/sheet" << i << ".xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>";
        out << "</Types>";
    }
    {
        ofstream out(temp / "_rels" / ".rels", ios::binary);
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            << "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            << "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
            << "</Relationships>";
    }
    {
        ofstream out(temp / "xl" / "workbook.xml", ios::binary);
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            << "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>";
        int sid = 1;
        for (int mode = CHARWISE; mode <= WORDS; mode++)
        {
            out << "<sheet name=\"" << xmlEscape(tokenizationName(mode)) << "\" sheetId=\"" << sid << "\" r:id=\"rId" << sid << "\"/>";
            sid++;
        }
        out << "</sheets></workbook>";
    }
    {
        ofstream out(temp / "xl" / "_rels" / "workbook.xml.rels", ios::binary);
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            << "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">";
        for (int i = 1; i <= 3; i++)
            out << "<Relationship Id=\"rId" << i << "\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet" << i << ".xml\"/>";
        out << "</Relationships>";
    }
    int sid = 1;
    for (int mode = CHARWISE; mode <= WORDS; mode++)
        writeWorksheetXml((temp / "xl" / "worksheets" / ("sheet" + to_string(sid++) + ".xml")).string(), rows, mode);
    fs::path outPath(outName);
    fs::path zipName = outPath;
    zipName.replace_extension(".zip");
    fs::remove(outPath);
    fs::remove(zipName);
    string cmd = "powershell -NoProfile -Command \"Compress-Archive -Path '" + temp.string() + "\\*' -DestinationPath '" + zipName.string() + "' -Force\"";
    int rc = system(cmd.c_str());

    if (rc == 0 && fs::exists(zipName))
        fs::rename(zipName, outPath);

    fs::remove_all(temp);
    return rc == 0 && fs::exists(outPath);
}

void compare(const vector<uint32_t>& v1, const vector<uint32_t>& v2)
{
    size_t n = v1.size();
	size_t safe_size = min(n, v2.size());
	cout << "Comparing vectors of size " << n << " and " << v2.size() << "\n";
    for (size_t i = 0; i < safe_size; i++)
    {
        if (v1[i] != v2[i])
		{
			cout << "Mismatch at index " << i << ": " << v1[i] << " != " << v2[i] << "\n";
			return;
		}
	}
}

void DeleteSimpleArtifacts(const string& baseName)
{
	vector<string> files = {
		baseName + ".dict",
		baseName + ".freq",
		baseName + ".code",
		baseName + ".code.tbl",
	};
	for (const auto& file : files)
	{
		if (std::filesystem::exists(file))
		{
			std::error_code ec;
			std::filesystem::remove(file, ec);
			if (ec) {
				std::cerr << "Warning: Could not delete " << file << ": " << ec.message() << "\n";
			}
		}
	}
}

void DeletePPMDArtifacts()
{
	// delete any .ppmd.7z files in the current directory
	for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
	{
		if (entry.is_regular_file() && entry.path().string().find(".ppmd") != std::string::npos)
		{
			std::error_code ec;
			std::filesystem::remove(entry.path(), ec);
			if (ec) {
				std::cerr << "Warning: Could not delete " << entry.path() << ": " << ec.message() << "\n";
			}
		}
	}
}

void printUsage()
{
    cout << "Usage: MyCompressor <command> <input> [mode] [compression_algorithm]\n";
    cout << "Commands:\n";
    cout << "  benchmark runs specified compression algorithm and mode\n";
    cout << "  benchmark-xlsx [files.csv] [benchmark.xlsx]\n";
    cout << "    Reads filenames from files.csv, runs all methods for modes 1-3,\n";
    cout << "    and writes an Excel workbook with one sheet per tokenization type.\n";
    cout << "Modes:\n";
    cout << "  1 - Character-wise tokenization\n";
    cout << "  2 - Space-based tokenization\n";
    cout << "  3 - Word-based tokenization\n";
    cout << "Compression algorithms:\n";
    for (int a = UNIFORM_TANS; a <= STATIC_ARITHMETIC; a++)
        cout << "  " << a << " - " << algorithmName(a) << "\n";
}
int main(int argc, char* argv[]) {
    int N = 50;
    if (argc < 2) {
        printUsage();
        return 0;
    }

    string cmd = argv[1];

    if (cmd == "benchmark-xlsx") {
        string csvName = argc >= 3 ? argv[2] : "files.csv";
        string reportName = argc >= 4 ? argv[3] : "benchmark.xlsx";
        vector<string> files = readFileListCsv(csvName);

        if (files.empty()) {
            cerr << "No filenames found in " << csvName << "\n";
            return 1;
        }

        vector<BenchmarkRecord> records;

        for (const string& input : files) {
            string text = readFile(input);
            if (text.empty()) {
                cerr << "Skipping empty or unreadable file: " << input << "\n";
                continue;
            }

            for (int mode = CHARWISE; mode <= WORDS; mode++) {
                cout << "Tokenizing " << input << " with " << tokenizationName(mode) << " tokenization...\n";
                auto tokens = tokenize(text, mode);

                for (int algorithm = UNIFORM_TANS; algorithm <= STATIC_ARITHMETIC; algorithm++) if (hasDecoder(algorithm)) {
                    cout << "  " << algorithmName(algorithm) << "..." << flush;

                    BenchmarkRecord rec;
                    rec.filename = input;
                    rec.mode = mode;
                    rec.algorithm = algorithm;

                    try {
                        IndexedTokens indexed = 
                            (algorithm == FORWARD_TANS || algorithm == ARITHMETIC_FORWARD) ? buildIndexLast(tokens) :
                            (algorithm == HUFFMAN_CANONICAL) ? buildIndexFreq(tokens) :
                            buildIndexSort(tokens);

                        string baseName = safeBaseName(input) + ".m" + to_string(mode) + ".a" + to_string(algorithm) + ".arc";
                        string codeFile = baseName + ".code";
                        auto cStart = chrono::high_resolution_clock::now();
                        for (int i = 0; i < N - 1; i++) encode(indexed, algorithm, codeFile);
                        EncodedData data = encode(indexed, algorithm, codeFile);
                        auto cEnd = chrono::high_resolution_clock::now();

                        rec.compressSeconds = chrono::duration<double>(cEnd - cStart).count() / N;
                        rec.codeBytes = compressedSize(data) + 8 + ((algorithm == HUFFMAN_CANONICAL || algorithm == ARITHMETIC_BACKWARD || algorithm == UNIFORM_TANS || algorithm == RANGED_TANS) ? 8 : 0); // add bitcount and original size of input or final state, if neccesary
                        writeCompressed(baseName, data, mode);

                        rec.dictPpmdBytes = compressWith7zPPMd(baseName + ".dict");
                        rec.freqPpmdBytes = (algorithm == FORWARD_TANS || algorithm == ARITHMETIC_BACKWARD) ? 0 : compressWith7zPPMd(baseName + ((algorithm == HUFFMAN_CANONICAL) ? ".code.tbl" : ".freq"));
                        rec.totalBytes = rec.codeBytes + rec.dictPpmdBytes + rec.freqPpmdBytes;

                        int readMode;
                        EncodedData readData = readCompressed(baseName, readMode);
                        auto dStart = chrono::high_resolution_clock::now();
                        for (int i = 0; i < N - 1; i++) decodeByAlgorithm(readData, baseName, algorithm);
                        vector<uint32_t> ids = decodeByAlgorithm(readData, baseName, algorithm);
                        auto dEnd = chrono::high_resolution_clock::now();
                        rec.decompressSeconds = chrono::duration<double>(dEnd - dStart).count() / N;
						
                        DeleteSimpleArtifacts(baseName);

                        cout << " done\n";
                    }
                    catch (const exception& e) {
                        cerr << " failed: " << e.what() << "\n";
                    }
                    catch (...) {
                        cerr << " failed with unknown error\n";
                    }

                    records.push_back(rec);
                }
            }
        }
		DeletePPMDArtifacts();
        if (!writeXlsxReport(reportName, records)) {
            cerr << "Could not create " << reportName << ". On Windows this requires PowerShell Compress-Archive.\n";
            return 1;
        }

        cout << "Report written to " << reportName << "\n";
        return 0;
    }

    if (cmd == "benchmark") {
        if (argc < 5) {
            cout << "Usage: MyCompressor benchmark <input> <mode> <compression_algorithm>\n";
            return 0;
        }
        string input = argv[2];
        int mode = stoi(argv[3]);
        int algorithm = stoi(argv[4]);

        string output = makeCompressedName(input);
        string codeFile = output + ".code";
        uint64_t textSize = 0;
        EncodedData data;
        IndexedTokens indexed;
        {
            string text = readFile(input);
            textSize = text.size();
            auto tokens = tokenize(text, mode);
            indexed = (algorithm == FORWARD_TANS || algorithm == ARITHMETIC_FORWARD) ? buildIndexLast(tokens) : (algorithm == HUFFMAN_CANONICAL) ? buildIndexFreq(tokens) : buildIndexSort(tokens);
        }
        auto t0 = chrono::high_resolution_clock::now();
            data = encode(indexed, algorithm, codeFile);
        auto t1 = chrono::high_resolution_clock::now();
        writeCompressed(output, data, mode);

        double compTime = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() / 1000.0;

        double decompTime = -1.0;
        if (hasDecoder(algorithm)) {
            int readMode;
            EncodedData readData = readCompressed(output, readMode);
            auto t2 = chrono::high_resolution_clock::now();
            vector<uint32_t> ids = decodeByAlgorithm(readData, output, algorithm);
            auto t3 = chrono::high_resolution_clock::now();
            string decOut = makeDecompressedName(output);
            writeFile(decOut, ids, readData.dictionary);
            decompTime = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count() / 1000.0;
            compare(indexed.ids, ids);
        }
        cout << "Benchmark results:\n";
        cout << "  Original size: " << textSize << " bytes\n";
        uint64_t codeBytes = compressedSize(data) + 8 + ((algorithm == HUFFMAN_CANONICAL || algorithm == ARITHMETIC_BACKWARD || algorithm == UNIFORM_TANS || algorithm == RANGED_TANS) ? 8 : 0); // add bitcount and original size of input or final state, if neccesary
        uint64_t dictPpmdBytes = compressWith7zPPMd(output + ".dict");
        uint64_t freqPpmdBytes = compressWith7zPPMd(output + ((algorithm == HUFFMAN_CANONICAL) ? "code.tbl" : ".freq"));
        cout << "  Code size: " << codeBytes << " bytes\n";
        cout << "  Dictionary size after 7-Zip PPMd -mx=9: " << dictPpmdBytes << " bytes\n";
        cout << "  " << ((algorithm == HUFFMAN_CANONICAL) ? "Code" : "Frequency") << " table size after 7 - Zip PPMd - mx = 9: " << freqPpmdBytes << " bytes\n";
        cout << "  Total size: " << (codeBytes + dictPpmdBytes + freqPpmdBytes) << " bytes\n";
        cout << "  Compression time: " << compTime << " s\n";
        if (decompTime >= 0.0) cout << "  Decompression time: " << decompTime << " s\n";
        else cout << "  Decompression time: N/A\n";
    }

    else {
        printUsage();
    }

    return 0;
}