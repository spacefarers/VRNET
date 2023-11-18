import run

pretrain_vars = ["160","320","6400"]

if __name__ == "__main__":
    starting_id = run.run_id
    for ind,var in enumerate(pretrain_vars):
        run.selected_var = var
        dataset_io = run.Dataset(run.root_data_dir, run.dataset, run.selected_var, run.scale)
        dataset_io.load()
        run.run_id = starting_id + ind