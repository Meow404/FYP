import random

start = int(input("Start : "));
num_of_kernels = int(input("Enter num of kernels : "));

kernel_file = open(r"kernels.txt","w")

kernel_file.write(str(num_of_kernels)+"\n")

for i in range(num_of_kernels):
	kernel_file.write(str(start+2*i)+"\n")
	for _ in range(start+2*i):
		kernel_file.write(",".join([str(random.randint(1,32)) for j in range(start+2*i)])+",\n")
