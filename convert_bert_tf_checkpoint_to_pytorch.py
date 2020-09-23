#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 23/09/2020 20:22 
@Author: XinZhi Yao 
"""


import logging
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional
import argparse
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert


## utils 代码部分

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.WARN)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name: Optional[str] = None) -> logging.Logger:

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(INFO)


def set_verbosity_warning():
    return set_verbosity(WARNING)


def set_verbosity_debug():
    return set_verbosity(DEBUG)


def set_verbosity_error():
    return set_verbosity(ERROR)


def disable_default_handler() -> None:

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)


