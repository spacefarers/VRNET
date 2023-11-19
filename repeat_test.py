import run
import numpy as np
from datetime import datetime
import time
import json
from pathlib import Path

test_epochs = (0, 50, 200)
test_amount = 10

formatted_time = datetime.now().strftime("%Y%b%d_%H%M%S")
PSNR_list = []
starting_id = run.run_id
repeated_test_log = {"test_epochs": test_epochs, "test_amount": test_amount, "starting_id": starting_id}


def save_log():
    Path(f"{run.experiments_dir}repeat_logs/").mkdir(parents=True, exist_ok=True)
    with open(f"{run.experiments_dir}repeat_logs/repeated_test_log_{formatted_time}.json", "w") as f:
        json.dump(repeated_test_log, f)


if __name__ == "__main__":
    for i in range(test_amount):
        run.run_id = starting_id + i
        start_time = time.time()
        PSNR, pretrain_time_cost, finetune1_time_cost, finetune2_time_cost = run.run(test_epochs)
        end_time = time.time()
        test_time_cost = end_time - start_time
        print(f"PSNR of test {run.run_id} is {PSNR}")
        print(f"Total time cost is {test_time_cost}")
        repeated_test_log[run.run_id] = {"PSNR": PSNR, "pretrain_time": pretrain_time_cost,
                                         "finetune1_time": finetune1_time_cost, "finetune2_time": finetune2_time_cost,
                                         "total_time": test_time_cost}
        PSNR_list.append(PSNR)
        save_log()

    print(f"Mean PSNR is {np.mean(PSNR_list)}")
    print(f"array:\n {PSNR_list}")
    repeated_test_log["mean_PSNR"] = np.mean(PSNR_list)
    repeated_test_log["PSNR_list"] = PSNR_list
    save_log()
