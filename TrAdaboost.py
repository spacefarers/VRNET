from dataset_io import Dataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot


def TrAdaboost(run_id=200, iters=5, tag="TrA"):
    print(f"Running {tag} {run_id}...")
    config.tags.append(tag)
    config.run_id = run_id

    dataset_io = Dataset(config.target_dataset, config.target_var, "all")
    if config.load_ensemble_model:
        config.ensemble_path = config.experiments_dir + f"{(run_id - 100):03d}/finetune2.pth"
        print("Loading ensemble model...")
        PSNR, _ = infer_and_evaluate(config.ensemble_path, write_to_file=False)
        print(f"Baseline PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    M = model.prep_model(model.Net())
    D = model.prep_model(model.D())
    T = train.Trainer(dataset_io, M, D)
    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        T.train(disable_jump=True if cycle > 1 else False)
        PSNR, PSNR_list = infer_and_evaluate(T.model, write_to_file=False if cycle != cycles else True, inference_dir=T.inference_dir, experiments_dir=T.experiment_dir)
        save_plot(PSNR, PSNR_list, T.experiment_dir, run_cycle=cycle)
        print(f"Cycle {cycle} PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(run)
