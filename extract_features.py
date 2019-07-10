from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

import numpy as np
import sentencepiece as spm
import six
import codecs
import collections
import tensorflow as tf

import xlnet
from model_utils import init_from_checkpoint, configure_tpu
from prepro_utils import preprocess_text, encode_ids
from run_classifier import InputExample, PaddingInputExample
from run_classifier import file_based_convert_examples_to_features, file_based_input_fn_builder

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(guid=unique_id, text_a=text_a, text_b=text_b, label=0.0))
            unique_id += 1
    return examples


def model_fn_builder():
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        bsz_per_core = tf.shape(features["input_ids"])[0]

        inp = tf.transpose(features["input_ids"], [1, 0])
        seg_id = tf.transpose(features["segment_ids"], [1, 0])
        inp_mask = tf.transpose(features["input_mask"], [1, 0])
        label = tf.reshape(features["label_ids"], [bsz_per_core])

        xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
        run_config = xlnet.create_run_config(is_training, True, FLAGS)

        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=inp,
            seg_ids=seg_id,
            input_mask=inp_mask)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # load pretrained models
        scaffold_fn = init_from_checkpoint(FLAGS)

        # Get a summary of the sequence using the last hidden state
        summary = xlnet_model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)

        # Get a sequence output
        seq_out = xlnet_model.get_sequence_output()

        batch_size = FLAGS.predict_batch_size

        predictions = {}
        # for i in range(batch_size):
        #     predictions['pooled_%d' % i] = summary[i]

        predictions['pooled'] = summary
        if FLAGS.use_tpu:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.do_predict:
        predict_dir = FLAGS.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    spiece_model_file = FLAGS.spiece_model_file
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spiece_model_file)

    label_list = [0.0]

    model_fn = model_fn_builder()

    run_config = configure_tpu(FLAGS)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    examples = read_examples(FLAGS.input_file)

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    #
    # Modified in XL: We also adopt the same mechanism for GPUs.
    while len(examples) % FLAGS.predict_batch_size != 0:
        examples.append(PaddingInputExample())

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    pred_file_base = "{}.len-{}.{}.predict.tf_record".format(
        spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
    pred_file = os.path.join(FLAGS.output_dir, pred_file_base)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp_model, text)

    # create a tf_record from examples
    file_based_convert_examples_to_features(
        examples, label_list, FLAGS.max_seq_length, tokenize_fn,
        pred_file)

    assert len(examples) % FLAGS.predict_batch_size == 0

    pred_input_fn = file_based_input_fn_builder(
        input_file=pred_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)

    with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file, "w")) as writer:
        for example_cnt, result in enumerate(estimator.predict(input_fn=pred_input_fn,
                                                               yield_single_examples=True,
                                                               checkpoint_path=FLAGS.predict_ckpt)):
            if example_cnt % 1000 == 0:
                tf.logging.info("Predicting submission for example_cnt: {}".format(example_cnt))
            output_json = collections.OrderedDict()
            output_json['pooled_%d' % example_cnt] = [round(float(x), 6) for x in result['pooled'].flat]
            writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("model_config_path")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("spiece_model_file")
    tf.app.run()
