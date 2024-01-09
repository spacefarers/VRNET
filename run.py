from dataset_io import Dataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot


def run(run_id=15, finetune1_epochs=20, finetune2_epochs=0, cycles=5, load_ensemble_model=False, tag="run", use_all_data=False, swap_source_target=False):
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
    target_dataset = config.target_dataset
    target_var = config.target_var
    if swap_source_target:
        target_dataset = config.source_dataset
        target_var = config.source_var
    dataset_io = Dataset(target_dataset, target_var, "all" if use_all_data else "train")
    if config.load_ensemble_model:
        config.ensemble_path = config.experiments_dir + f"{(run_id - 100):03d}/finetune2.pth"
        print("Loading ensemble model...")
        PSNR, _ = infer_and_evaluate(config.ensemble_path, write_to_file=False)
        print(f"Baseline PSNR: {PSNR}")
    M = model.prep_model(model.Net())
    D = model.prep_model(model.D())
    T = train.Trainer(dataset_io, M, D)
    source_eval = Dataset(config.source_dataset, config.source_var, "all")
    for cycle in range(1, cycles + 1):
        print(f"Cycle {cycle}/{cycles}:")
        T.train(disable_jump=True if cycle > 1 else False)
        target_PSNR, _ = infer_and_evaluate(T.model)
        source_PSNR, _ = infer_and_evaluate(T.model, data=source_eval)
    config.set_status("Succeeded")


if __name__ == "__main__":
    fire.Fire(run)
