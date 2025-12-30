import math

def log_sigma_for_weight(W):
    return -0.5 * math.log(2 * W)

def main():
    while True:
        try:
            user_input = input("Enter target scaling (positive number), or 'q' to quit: ")
            if user_input.lower() == 'q':
                print("Exiting.")
                break
            W = float(user_input)
            if W <= 0:
                print("Please enter a positive number.")
                continue
            log_sigma = log_sigma_for_weight(W)
            print(f"Target scaling: {W}")
            print(f"Corresponding log_sigma: {log_sigma:.4f}\n")
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit.\n")

if __name__ == "__main__":
    main()
