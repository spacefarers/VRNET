from dataset_io import Dataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot


def run(run_id=100, finetune1_epochs=20, finetune2_epochs=0, cycles=1, load_ensemble_model=False, tag="run"):
    print(f"Running {tag} {run_id}...")
    if tag == "run":
        config.lr = (1e-6, 4e-6)
        config.tags.append("barebone" if cycles == 1 else "cycles")
    elif tag == "EN-FT":
        config.lr = (1e-6, 4e-6)
        config.tags.append("EN-FT")
    else:
        raise Exception("Unknown tag")
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    config.load_ensemble_model = load_ensemble_model
    dataset_io = Dataset(config.dataset, config.target_var, "train")
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
        pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, _, _ = T.train(
            disable_jump=True if cycle > 1 else False)
        PSNR, PSNR_list = infer_and_evaluate(T.model, write_to_file=False if cycle != cycles else True, inference_dir=T.inference_dir, experiments_dir=T.experiment_dir)
        save_plot(PSNR, PSNR_list, T.experiment_dir, run_cycle=cycle)
        print(f"Cycle {cycle} PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    config.log({"status": 4})


if __name__ == "__main__":
    fire.Fire(run)
