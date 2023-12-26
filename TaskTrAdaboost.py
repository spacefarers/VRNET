from dataset_io import Dataset
import config
import model
import train
import fire


def TaskTrAdaboost(run_id=500, iters=20, tag="TTrA"):
    print(f"Running {tag} {run_id}...")
    config.run_id = run_id
    dataset_io = Dataset(config.dataset,config.target_var)
    # dataset_io = Dataset("default")
    config.ensemble_path = config.experiments_dir + f"{(run_id - 100):03d}/finetune2.pth"
    if config.load_ensemble_model:
        print("Loading ensemble model...")
        for config.domain_backprop in [False, True]:
            M = model.Net()
            D = model.D()
            M = model.prep_model(M)
            D = model.prep_model(D)
            T = train.Trainer(dataset_io, M, D)
            try:
                T.load_model(config.ensemble_path)
            except RuntimeError:
                continue
            break
        PSNR, _ = T.inference(write_to_file=False)
        T.save_plot()
        print(f"Baseline PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        T.run_cycle = cycle
        pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, _, _ = T.train(
            disable_jump=True if cycle > 1 else False)
        PSNR, _ = T.inference(write_to_file=False if cycle != cycles else True, disable_jump=True)
        T.save_plot()
        print(f"Cycle {cycle} PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    config.log({"status": 4})


if __name__ == "__main__":
    fire.Fire(run)
