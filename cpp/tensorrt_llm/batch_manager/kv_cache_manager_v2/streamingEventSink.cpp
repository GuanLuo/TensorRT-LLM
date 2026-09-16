/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/streamingEventSink.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/page.h"
#include "tensorrt_llm/common/logger.h"

#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

class MessagePackWriter
{
public:
    void packArraySize(size_t size)
    {
        if (size <= 15)
        {
            mPayload.push_back(static_cast<uint8_t>(0x90U | size));
        }
        else if (size <= std::numeric_limits<uint16_t>::max())
        {
            mPayload.push_back(0xDCU);
            packBigEndian(static_cast<uint16_t>(size));
        }
        else
        {
            if (size > std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error("MessagePack array is too large");
            }
            mPayload.push_back(0xDDU);
            packBigEndian(static_cast<uint32_t>(size));
        }
    }

    void packMapSize(size_t size)
    {
        if (size <= 15)
        {
            mPayload.push_back(static_cast<uint8_t>(0x80U | size));
            return;
        }
        throw std::overflow_error("Streaming event map is too large");
    }

    void packString(std::string_view value)
    {
        if (value.size() <= 31)
        {
            mPayload.push_back(static_cast<uint8_t>(0xA0U | value.size()));
        }
        else if (value.size() <= std::numeric_limits<uint8_t>::max())
        {
            mPayload.push_back(0xD9U);
            mPayload.push_back(static_cast<uint8_t>(value.size()));
        }
        else
        {
            throw std::overflow_error("Streaming event string is too large");
        }
        mPayload.insert(mPayload.end(), value.begin(), value.end());
    }

    void packInt(int64_t value)
    {
        mPayload.push_back(0xD3U);
        packBigEndian(static_cast<uint64_t>(value));
    }

    void packDouble(double value)
    {
        static_assert(sizeof(value) == sizeof(uint64_t));
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        mPayload.push_back(0xCBU);
        packBigEndian(bits);
    }

    void packNil()
    {
        mPayload.push_back(0xC0U);
    }

    [[nodiscard]] std::vector<uint8_t> takePayload()
    {
        return std::move(mPayload);
    }

private:
    template <typename T>
    void packBigEndian(T value)
    {
        static_assert(std::is_unsigned_v<T>);
        for (size_t shift = sizeof(T); shift > 0; --shift)
        {
            mPayload.push_back(static_cast<uint8_t>(value >> ((shift - 1) * 8U)));
        }
    }

    std::vector<uint8_t> mPayload;
};

void packIntArray(MessagePackWriter& writer, std::vector<int64_t> const& values)
{
    writer.packArraySize(values.size());
    for (int64_t value : values)
    {
        writer.packInt(value);
    }
}

void packTokenArray(MessagePackWriter& writer, std::vector<TokenId> const& values)
{
    writer.packArraySize(values.size());
    for (TokenId value : values)
    {
        writer.packInt(value);
    }
}

void packStoredEvent(MessagePackWriter& writer, StreamingBlockStoredData const& event, int blockSize)
{
    writer.packMapSize(8);
    writer.packString("type");
    writer.packString("BlockStored");
    writer.packString("block_hashes");
    packIntArray(writer, event.blockHashes);
    writer.packString("parent_block_hash");
    if (event.parentBlockHash.has_value())
    {
        writer.packInt(*event.parentBlockHash);
    }
    else
    {
        writer.packNil();
    }
    writer.packString("token_ids");
    packTokenArray(writer, event.tokenIds);
    writer.packString("block_size");
    writer.packInt(blockSize);
    writer.packString("lora_id");
    writer.packNil();
    writer.packString("medium");
    writer.packString("GPU");
    writer.packString("lora_name");
    writer.packNil();
}

void packRemovedEvent(MessagePackWriter& writer, StreamingBlockRemovedData const& event)
{
    writer.packMapSize(3);
    writer.packString("type");
    writer.packString("BlockRemoved");
    writer.packString("block_hashes");
    packIntArray(writer, event.blockHashes);
    writer.packString("medium");
    writer.packString("GPU");
}

StreamingSerializedBatch serializeBatch(
    std::vector<StreamingEventData> const& events, int blockSize, double timestamp, int dataParallelRank)
{
    MessagePackWriter writer;
    writer.packArraySize(3);
    writer.packDouble(timestamp);
    writer.packArraySize(events.size());
    for (auto const& event : events)
    {
        std::visit(
            [&writer, blockSize](auto const& data)
            {
                using T = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<T, StreamingBlockStoredData>)
                {
                    packStoredEvent(writer, data, blockSize);
                }
                else
                {
                    packRemovedEvent(writer, data);
                }
            },
            event);
    }
    writer.packInt(dataParallelRank);
    return StreamingSerializedBatch{writer.takePayload(), events.size()};
}

} // namespace

StreamingEventSink::StreamingEventSink(int tokensPerBlock, int maxEntries)
    : mTokensPerBlock(tokensPerBlock)
    , mMaxEntries(maxEntries)
{
    if (mTokensPerBlock <= 0)
    {
        throw std::invalid_argument("tokensPerBlock must be positive");
    }
    if (mMaxEntries <= 0)
    {
        throw std::invalid_argument("maxEntries must be positive");
    }
}

void StreamingEventSink::setTargetLifeCycle(LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mTargetLifeCycle = lifeCycle;
}

std::vector<StreamingEventData> StreamingEventSink::drainIterationEvents()
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto events = std::move(mPendingEvents);
    mPendingEvents.clear();
    mPendingEntries = 0;
    return events;
}

std::optional<StreamingSerializedBatch> StreamingEventSink::drainSerializedIteration(
    double timestamp, int dataParallelRank)
{
    auto events = drainIterationEvents();
    if (events.empty())
    {
        return std::nullopt;
    }
    return serializeBatch(events, mTokensPerBlock, timestamp, dataParallelRank);
}

StreamingEventStats StreamingEventSink::getStats() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mStats;
}

void StreamingEventSink::addStoredBlock(Block const& block)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addStoredBlockUnlocked(block);
}

void StreamingEventSink::addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle != *mTargetLifeCycle)
    {
        ++mStats.nonTargetLifeCyclesIgnored;
        return;
    }
    addStoredBlockUnlocked(block);
}

void StreamingEventSink::addRemovedBlock(Digest const& blockKey)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addRemovedBlockUnlocked(blockKey);
}

void StreamingEventSink::addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle != *mTargetLifeCycle)
    {
        ++mStats.nonTargetLifeCyclesIgnored;
        return;
    }
    addRemovedBlockUnlocked(blockKey);
}

void StreamingEventSink::addCacheLevelUpdated(Digest const&, CacheLevel, CacheLevel, LifeCycleId)
{
    // The streaming protocol currently tracks radix-tree residency, not cache-tier movement.
}

void StreamingEventSink::addStoredBlockUnlocked(Block const& block)
{
    if (!mTargetLifeCycle.has_value() || *mTargetLifeCycle >= block.storage.size())
    {
        return;
    }
    auto const* page = block.getPage(*mTargetLifeCycle);
    if (page == nullptr)
    {
        return;
    }
    if (!block.isFull() || page->numTokensInBlock < static_cast<int>(block.tokens.size()))
    {
        ++mStats.partialBlocksSuppressed;
        return;
    }
    if (mStoredBlocks.count(block.key) != 0)
    {
        return;
    }
    for (auto const& token : block.tokens)
    {
        if (token.isDigest())
        {
            ++mStats.multimodalBlocksSuppressed;
            return;
        }
    }
    if (!reserveEntryUnlocked())
    {
        return;
    }
    if (block.prev == nullptr)
    {
        throw std::logic_error("Cannot publish an orphan KV cache block");
    }

    int64_t const blockHash = wireHash(block.key);
    std::optional<int64_t> parentHash;
    if (block.prev->type() == NodeBase::Type::kBLOCK)
    {
        parentHash = wireHash(static_cast<Block const*>(block.prev)->key);
    }

    std::vector<TokenId> tokenIds;
    tokenIds.reserve(block.tokens.size());
    for (auto const& token : block.tokens)
    {
        tokenIds.push_back(token.tokenId());
    }

    mStoredBlocks.emplace(block.key, blockHash);
    if (!mPendingEvents.empty())
    {
        auto* stored = std::get_if<StreamingBlockStoredData>(&mPendingEvents.back());
        if (stored != nullptr && !stored->blockHashes.empty() && parentHash.has_value()
            && stored->blockHashes.back() == *parentHash)
        {
            stored->blockHashes.push_back(blockHash);
            stored->tokenIds.insert(stored->tokenIds.end(), tokenIds.begin(), tokenIds.end());
            ++mStats.storedBlocks;
            return;
        }
    }
    mPendingEvents.emplace_back(StreamingBlockStoredData{{blockHash}, parentHash, std::move(tokenIds)});
    ++mStats.storedBlocks;
}

void StreamingEventSink::addRemovedBlockUnlocked(Digest const& blockKey)
{
    auto const stored = mStoredBlocks.find(blockKey);
    if (stored == mStoredBlocks.end())
    {
        return;
    }
    int64_t const blockHash = stored->second;
    mStoredBlocks.erase(stored);
    addRemovedHashUnlocked(blockHash);
}

void StreamingEventSink::addRemovedHashUnlocked(int64_t blockHash)
{
    if (!mPendingEvents.empty())
    {
        auto* removed = std::get_if<StreamingBlockRemovedData>(&mPendingEvents.back());
        if (removed != nullptr)
        {
            removed->blockHashes.push_back(blockHash);
            ++mStats.removedBlocks;
            return;
        }
    }
    mPendingEvents.emplace_back(StreamingBlockRemovedData{{blockHash}});
    ++mStats.removedBlocks;
}

bool StreamingEventSink::reserveEntryUnlocked()
{
    if (mPendingEntries < mMaxEntries)
    {
        ++mPendingEntries;
        return true;
    }
    ++mStats.droppedEvents;
    int64_t const dropped = mStats.droppedEvents;
    if (dropped == 1 || (dropped & (dropped - 1)) == 0)
    {
        TLLM_LOG_WARNING(
            "Dropping streaming KV events because the per-iteration safety cap was exceeded; "
            "dropped_events=%" PRId64,
            dropped);
    }
    return false;
}

int64_t StreamingEventSink::wireHash(Digest const& digest)
{
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i)
    {
        value = (value << 8U) | std::to_integer<uint8_t>(digest[i]);
    }
    uint64_t constexpr kSignedMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (value <= kSignedMax)
    {
        return static_cast<int64_t>(value);
    }
    return -static_cast<int64_t>(~value) - 1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
