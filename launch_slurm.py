# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import re
import pickle
import uuid
from pathlib import Path
import submitit
import train_submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for StyleGAN3 slurm training", parents=[train_submitit.get_args_parser()])
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--nodelist", default='grogu-1-29', type=str, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2880, type=int, help="Duration of the job")
    # parser.add_argument("--port", default=12343, type=int, help="distributed port")
    parser.add_argument("--partition", default="junyanlong", type=str, help="Partition where to submit")
    return parser.parse_args()


class Trainer(object):
    def __init__(self, training_args):
        self.training_args = training_args

    def __call__(self):
        import train_submitit
        train_submitit.main(self.training_args)

    def checkpoint(self):
        import os
        import submitit

        opts = self.training_args
        run_dir = os.path.join(opts.outdir, opts.exp_name)
        opts.resume = os.path.join(run_dir , f'network-snapshot-latest.pkl')

        with open(os.path.join(run_dir, 'latest_kimg.pkl'), 'rb') as f:
            resume_kimg = pickle.load(f)
        opts.resume_kimg = resume_kimg

        print("Requeuing...")
        empty_trainer = type(self)(opts)
        return submitit.helpers.DelayedSubmission(empty_trainer)

 
def main():
    args = parse_args()
    run_dir = os.path.join(args.outdir, args.exp_name)
    executor = submitit.AutoExecutor(folder=run_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.gpus
    nodes = args.nodes
    nodelist = args.nodelist
    timeout_min = args.timeout
    partition = args.partition
  
    executor.update_parameters(
        mem_gb=32 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        cpus_per_task=16,
        tasks_per_node=1,  # one task per GPU
        nodes=nodes,
        # slurm_constraint='rtx6000',
        slurm_additional_parameters={"nodelist": args.nodelist},
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
    )

    # executor.update_parameters(name=args.data.split('/')[-1])

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.outdir}")
    # job._interrupt(timeout=True)


if __name__ == "__main__":
    main()
