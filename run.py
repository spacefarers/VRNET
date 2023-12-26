from dataset_io import Dataset
import config
import model
import train
import fire
from inference import infer_and_evaluate


def run(run_id=200, finetune1_epochs=5, finetune2_epochs=0, cycles=1, load_ensemble_model=False, tag="run"):
    print(f"Running {tag} {run_id}...")
    if tag == "run":
        config.lr = (1e-4, 4e-4)
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
