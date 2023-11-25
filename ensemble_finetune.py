from dataset_io import Dataset
import model
import os
import train
import fire
import config


def run(run_id=102, finetune1_epochs=10, finetune2_epochs=50):
    config.load_ensemble_model = True
    config.run_id = run_id
    config.finetune1_epochs = finetune1_epochs
    config.finetune2_epochs = finetune2_epochs
    dataset_io = Dataset("640")
    dataset_io.load()
    M = model.Net()
    D = model.D()
    M = model.prep_model(M)
    D = model.prep_model(D)
    T = train.Trainer(dataset_io, M, D)
    if config.load_ensemble_model:
        T.load_model(config.ensemble_path)
    pretrain_time_cost, finetune1_time_cost, finetune2_time_cost, _, _ = T.train(disable_jump=True)
    PSNR, _ = T.inference()
    T.save_plot()
    # T.increase_framerate(interval,"finetune2")
    return PSNR, pretrain_time_cost, finetune1_time_cost, finetune2_time_cost


if __name__ == "__main__":
    fire.Fire(run)
