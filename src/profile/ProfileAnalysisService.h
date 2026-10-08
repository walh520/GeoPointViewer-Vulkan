// 地质剖面后台服务声明：在独立低优先级线程中读取 LOD0 并生成专业剖面统计。
#pragma once

#include "ProfileTypes.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

namespace gpv
{

class SeismicCacheReader;

class ProfileAnalysisService
{
public:
    ProfileAnalysisService() = default;
    ~ProfileAnalysisService();

    ProfileAnalysisService(const ProfileAnalysisService&) = delete;
    ProfileAnalysisService& operator=(const ProfileAnalysisService&) = delete;

    static bool RunContractCheck(const std::filesystem::path& cacheDirectory);

    void Start(const SeismicCacheReader& cacheReader);
    void Stop();
    std::uint64_t PrepareViewport(const Bounds2D& viewport);
    std::uint64_t Analyze(
        const Bounds2D& viewport,
        const std::vector<ProfileWorldPoint>& polyline,
        double halfWidth,
        std::uint32_t outputBinCount = 2048);
    void Cancel();
    void SetRenderBusy(bool busy);
    std::optional<ProfileResult> PollCompleted();

    ProfileTaskState State() const;

private:
    void WorkerMain(std::stop_token stopToken);
    ProfileResult Execute(const ProfileRequest& request, std::stop_token stopToken);
    bool ShouldCancel(std::uint64_t requestId, std::stop_token stopToken) const;
    bool WaitForRenderBudget(std::uint64_t requestId, std::stop_token stopToken);

    const SeismicCacheReader* cacheReader_ = nullptr;
    std::jthread worker_;
    mutable std::mutex mutex_;
    std::condition_variable_any condition_;
    std::optional<ProfileRequest> pendingRequest_;
    std::optional<ProfileResult> completedResult_;
    std::atomic<std::uint64_t> latestRequestId_{ 0 };
    std::atomic<ProfileTaskState> state_{ ProfileTaskState::Idle };
    std::atomic<bool> renderBusy_{ false };
};

} // namespace gpv
