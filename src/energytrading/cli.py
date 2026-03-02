import argparse
import sys
from energytrading.data.synthetic import generate_synthetic_power_data

def main():
    parser = argparse.ArgumentParser(description="EnergyTrading Quant CLI")
    subparsers = parser.add_subparsers(dest="command")

    sim_parser = subparsers.add_parser("simulate", help="Simulate synthetic power prices")
    sim_parser.add_parser("run-api", help="Start the FastAPI server")

    args = parser.parse_args()

    if args.command == "simulate":
        df = generate_synthetic_power_data(days=30)
        print("Generated Synthetic Data (First 5 hours):")
        print(df.head())
    elif args.command == "run-api":
        import uvicorn
        print("Starting FastAPI server...")
        uvicorn.run("energytrading.api.main:app", host="0.0.0.0", port=8000)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()