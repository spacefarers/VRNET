import model
from dataset_io import Dataset, MixedDataset
import train
import time
import config
import json
from fire import Fire


def ensemble_training(run_id=132, finetune1_epochs=5, finetune2_epochs=5, ensemble_epochs=15):
    print(f"Running ensemble training {run_id}...")
    config.tags.append("ensemble_training")
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    config.domain_backprop = False
    datasets = {}
    config.lr = (1e-4, 4e-4)
    models = {}

    training_logs = {}
    for cycle in range(ensemble_epochs):
        config.ensemble_iter = cycle
        start_time = time.time()
        for pretrain_var in config.pretrain_vars:
            pretrain_dataset = Dataset(pretrain_var, train_all_data=True)
            M = model.Net()
            D = model.D()
            M = model.prep_model(M)
            D = model.prep_model(D)
            T = train.Trainer(pretrain_dataset, M, D)
            _, _, _, M, D = T.train(disable_jump=True if cycle > 0 else False)
            models[pretrain_var] = (M, D)
        end_time = time.time()
        print(f"Training Time cost: {end_time - start_time}")
        print("Evaluating...")
        T = train.Trainer(datasets[config.target_var], M, D)
        PSNR, _ = T.inference(write_to_file=True, disable_jump=True)
        config.log({"PSNR": PSNR})
        T.save_plot()
        print(f"Cycle {cycle} PSNR: {PSNR}")
        training_logs[cycle] = PSNR
        with open(f"{config.experiments_dir}{config.run_id:03d}/ensemble_training.json", "w") as f:
            json.dump(training_logs, f, indent=4)


if __name__ == "__main__":
    Fire(ensemble_training)
