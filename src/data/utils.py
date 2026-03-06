import math
from typing import Generator, List

import torch
import torch.nn.functional as F

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


def collate_segments_to_batch(segments: List[Segment]) -> Batch:
    has_act = segments[0].act is not None and segments[0].act.numel() > 0
    obs = torch.stack([s.obs for s in segments])
    act = torch.stack([s.act for s in segments]) if has_act else None
    rew = torch.stack([s.rew for s in segments])
    end = torch.stack([s.end for s in segments])
    trunc = torch.stack([s.trunc for s in segments])
    mask_padding = torch.stack([s.mask_padding for s in segments])
    return Batch(obs, act, rew, end, trunc, mask_padding, [s.info for s in segments], [s.id for s in segments])


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    pad_len_right = max(0, segment_id.stop - len(episode))
    pad_len_left = max(0, -segment_id.start)
    assert pad_len_right == pad_len_left == 0 or should_pad

    def pad(x):
        right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
        return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)
    mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

    has_act = episode.act is not None and episode.act.numel() > 0
    act = pad(episode.act[start:stop]) if has_act else None

    return Segment(
        pad(episode.obs[start:stop]),
        act,
        pad(episode.rew[start:stop]),
        pad(episode.end[start:stop]),
        pad(episode.trunc[start:stop]),
        mask_padding,
        info=episode.info,
        id=SegmentId(segment_id.episode_id, start, stop),
    )


class DatasetTraverser:
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return math.ceil(
            sum(
                [
                    math.ceil(self.dataset.lengths[episode_id] / self.chunk_size)
                    - int(self.dataset.lengths[episode_id] % self.chunk_size == 1)
                    for episode_id in range(self.dataset.num_episodes)
                ]
            )
            / self.batch_num_samples
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            for i in range(math.ceil(len(episode) / self.chunk_size)):
                start = i * self.chunk_size
                stop = (i + 1) * self.chunk_size
                segment = make_segment(
                    episode,
                    SegmentId(episode_id, start, stop),
                    should_pad=True,
                )
                if self.dataset._dataset_full_res is not None and "original_file_id" in episode.info:
                    segment_id_full_res = SegmentId(episode.info["original_file_id"], start, stop)
                    segment.info["full_res"] = self.dataset._dataset_full_res[segment_id_full_res].obs
                chunks.append(segment)
            if chunks and chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)

