import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader


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
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
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
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


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
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
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
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
