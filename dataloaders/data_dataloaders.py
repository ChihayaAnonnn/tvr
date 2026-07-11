import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.hard_negative_sampler import HardNegativeDistributedBatchSampler


def _configure_video_worker(_worker_id):
    torch.set_num_threads(1)
    try:
        import cv2
    except ImportError:
        return
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)


def _video_loader_kwargs(args):
    workers = int(args.num_thread_reader)
    kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_configure_video_worker,
        )
    return kwargs


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        strategy=args.strategy,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msrvtt_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
        return_sample_index=getattr(args, "use_explicit_hard_negative_loss", False),
        return_hard_negative=getattr(args, "use_explicit_hard_negative_loss", False),
        hard_negative_path=getattr(args, "hard_negative_path", ""),
        split_manifest_path=args.split_manifest,
        tqfs_cache_dir=getattr(args, "tqfs_cache_dir", ""),
    )

    local_batch_size = args.batch_size // args.n_gpu
    if getattr(args, "use_hard_negative_packing", False):
        train_sampler = HardNegativeDistributedBatchSampler(
            msrvtt_dataset,
            hard_negative_path=getattr(args, "hard_negative_path", ""),
            batch_size=local_batch_size,
            num_replicas=getattr(args, "world_size", args.n_gpu),
            rank=getattr(args, "rank", 0),
            seed=getattr(args, "hard_negative_pack_seed", 42),
            drop_last=True,
            shuffle=True,
        )
        dataloader = DataLoader(
            msrvtt_dataset,
            batch_sampler=train_sampler,
            **_video_loader_kwargs(args),
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
        dataloader = DataLoader(
            msrvtt_dataset,
            batch_size=local_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            **_video_loader_kwargs(args),
        )

    return dataloader, len(msrvtt_dataset), train_sampler

def _build_msrvtt_eval_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        **_video_loader_kwargs(args),
    )


def dataloader_msrvtt_val(args, tokenizer, subset="val"):
    msrvtt_valset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msrvtt_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
        tqfs_cache_dir=getattr(args, "tqfs_cache_dir", ""),
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )
    return _build_msrvtt_eval_loader(msrvtt_valset, args), len(msrvtt_valset)


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.test_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msrvtt_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
        tqfs_cache_dir=getattr(args, "tqfs_cache_dir", ""),
        multi_sentence_per_video=False,
    )
    return _build_msrvtt_eval_loader(msrvtt_testset, args), len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msvd_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **_video_loader_kwargs(args),
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msvd_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
    )
    dataloader_msvd = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        **_video_loader_kwargs(args),
    )
    return dataloader_msvd, len(msvd_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {
    "train": dataloader_msrvtt_train,
    "val": dataloader_msrvtt_val,
    "test": dataloader_msrvtt_test,
}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
