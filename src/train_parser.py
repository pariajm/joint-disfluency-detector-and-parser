import json
import argparse
import numpy as np
from main import run_train, make_hparams

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to the job dir')
parser.add_argument('--model-path', type=str, help='path to the model dir', default=None)
parser.add_argument('--eval-path', type=str, help='path to the eval dir', default=None)
args = parser.parse_args()


def run_self_attentive_parser(run_config_filename, model_path=None, eval_out_path=None):
	random_config = dict()
	random_config['results_path'] = eval_out_path
	random_config['model_path_base'] = model_path
	random_config['numpy_seed'] = np.random.randint(1, 40000)
	with open(run_config_filename, 'r') as fp:
		random_config.update(json.loads(fp.read()))


	args = argparse.Namespace()
	vars(args).update(random_config)

	hparams = make_hparams()
	hparams.set_from_args(args)
	run_train(args, hparams)

if __name__ == '__main__':
	run_self_attentive_parser(args.config, args.model_path, args.eval_path)

