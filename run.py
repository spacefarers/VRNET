from dataset_io import Dataset
import config
import model
import train
import fire


def run(run_id=101, finetune1_epochs=10, finetune2_epochs=0, cycles=1, load_ensemble_model=False, tag="run"):
    print(f"Running {tag} {run_id}...")
    if tag == "run":
        lr = (1e-4, 4e-4)
        config.tags.append("barebone" if cycles == 1 else "cycles")
    elif tag == "EN-FT":
        lr = (1e-6, 4e-6)
        config.tags.append("EN-FT")
    else:
        raise Exception("Unknown tag")
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    config.load_ensemble_model = load_ensemble_model
    dataset_io = Dataset("640")
    # dataset_io = Dataset("default")
    M = model.Net()
    D = model.D()
    M = model.prep_model(M)
    D = model.prep_model(D)
    T = train.Trainer(dataset_io, M, D)
    config.ensemble_path = config.experiments_dir + f"{(run_id - 100):03d}/finetune2.pth"
    if config.load_ensemble_model:
        print("Loading ensemble model...")
        T.load_model(config.ensemble_path)
        PSNR, _ = T.inference(write_to_file=False)
        T.save_plot()
        print(f"Baseline PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        T.run_cycle = cycle
        pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, _, _ = T.train(
            disable_jump=True if cycle > 1 else False)
        PSNR, _ = T.inference(write_to_file=False if cycle != cycles else True)
        T.save_plot()
        print(f"Cycle {cycle} PSNR: {PSNR}")
        config.log({"PSNR": PSNR})
    config.log({"status": 4})


if __name__ == "__main__":
    fire.Fire(run)
