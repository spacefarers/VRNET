from dataset_io import Dataset
import config
import model
import train
import fire
from inference import infer_and_evaluate, save_plot


def DomainAdaptation(run_id=200, finetune1_epochs=20, finetune2_epochs=0, cycles=1, tag="DA"):
    config.domain_backprop = True
    print(f"Running {tag} {run_id}...")
    config.lr = (1e-6, 4e-6)
    config.tags.append(tag)
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    dataset_io = Dataset(config.target_dataset, config.target_var, "train")
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
    config.run["status"] = "Succeeded"


if __name__ == "__main__":
    fire.Fire(DomainAdaptation)
