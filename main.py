"""Entry point for the program"""

import logging

from src.args import get_args
from src.utils import initialize_accelerator, setup_logging

logger = logging.getLogger(__name__)

# pylint: disable=C0415


def main_accelerate(args):
    """Main function of the program"""

    from src.train_accelerate import Trainer

    accelertor = initialize_accelerator(args)

    if accelertor.is_local_main_process:
        setup_logging()

    trainer = Trainer(args, accelertor)

    trainer.train()
    logger.info("Finish Training")


def main_xla(args):
    """Main function of the program"""

    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    import torch_xla.runtime as xr

    import wandb
    from src.train_xla import Trainer

    xr.use_spmd()
    xp.start_server(9012)

    if xm.is_master_ordinal():
        xr.initialize_cache("model_cache", readonly=False)
    else:
        xr.initialize_cache("model_cache", readonly=True)

    xm.rendezvous("sync")

    if xm.is_master_ordinal(local=False) and xr.host_index() == 0:
        if xm.is_master_ordinal(local=False) and xr.host_index() == 0:
            setup_logging()
            wandb.init(
                entity=args.entity,
                name=f"{args.run_name}_{args.model.split('/')[-1]}{'(rand)' if args.random_init else ''}_{args.dataset}_{args.start_id}",
                config=args.__dict__,
                project=args.project,
                resume=args.resume,
                id=args.start_id,
            )

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    _args = get_args()

    if _args.xla:
        main_xla(_args)
    else:
        main_accelerate(_args)
