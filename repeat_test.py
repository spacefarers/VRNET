import main
import numpy as np

test_epochs = (0, 10, 50)
test_amount = 10

if __name__ == "__main__":
	PSNR_list = []
	starting_id = main.run_id
	for i in range(test_amount):
		main.run_id = starting_id + i
		PSNR = main.run(test_epochs)
		print(f"PSNR of test {main.run_id} is {PSNR}")
		PSNR_list.append(PSNR)
	print(f"Mean PSNR is {np.mean(PSNR_list)}")
	print(f"array:\n {PSNR_list}")
