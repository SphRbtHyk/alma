"""Scripts for benchmarking execution speed per algorithm.
"""
import time
import pandas as pd
from alma.collation_algorithms.collation import Collation
import random


def generate_test_verses(num_verses=100, tokens_per_verse=10):
    """
    Generate synthetic test verses for benchmarking
    """
    # Simple Latin-like vocabulary for testing
    vocabulary = [
        "et", "in", "de", "ad", "per", "cum", "ex", "ab", "super", "sub",
        "dominus", "deus", "christus", "spiritus", "sanctus", "ecclesia",
        "homo", "vir", "mulier", "pater", "mater", "filius", "frater",
        "cor", "anima", "corpus", "sanguis", "verbum", "vita", "mors",
        "caelum", "terra", "mundus", "tempus", "dies", "nox", "lux",
        "amor", "pax", "gratia", "fides", "spes", "caritas", "veritas",
        "facere", "dicere", "venire", "ire", "videre", "audire"
    ]

    verses_a = []
    verses_b = []

    for _ in range(num_verses):
        # Generate base verse
        base_verse = " ".join(random.choices(vocabulary, k=tokens_per_verse))

        # Create a variation with some words swapped/added/removed
        words = base_verse.split()
        if len(words) > 3:
            # Swap two random words to create variation
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]

        variant_verse = " ".join(words)

        verses_a.append(base_verse)
        verses_b.append(variant_verse)

    return verses_a, verses_b


def benchmark_algorithm(algorithm_config, verses_a, verses_b, algorithm_name):
    """
    Benchmark a single alignment algorithm
    """
    times = []

    for verse_a, verse_b in zip(verses_a, verses_b):
        try:
            # Create collator with current configuration
            collator = Collation(**algorithm_config)

            # Time the collation
            start_time = time.time()
            collation_matches = collator.collate("A", verse_a, "B", verse_b)
            end_time = time.time()

            times.append(end_time - start_time)

        except Exception as e:
            print(f"Error with {algorithm_name}: {e}")
            times.append(float('nan'))  # Mark as failed

    # Calculate average time (ignoring failed runs)
    valid_times = [t for t in times if not pd.isna(t)]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        success_rate = len(valid_times) / len(times) * 100
    else:
        avg_time = float('nan')
        success_rate = 0

    return avg_time, success_rate, len(valid_times)


def run_benchmark():
    """
    Main benchmarking function
    """
    # Define the algorithms to test
    algorithms = {
        "collatex": {
            "collation_model": "collatex"
        },
        "pos_latin": {
            "collation_model": "pos",
            "model_language": "latin"
        },
        "simalign_a": {
            "collation_model": "simalign",
            "matching_methods": "a",
            "simalign_model": "bert-base-multilingual-cased"
        },
        "alma_bert": {
            "collation_model": "alma",
            "spacy_model": "la_core_web_lg",
            "similarity_threshold": 1,
            "embedding_model": "google-bert/bert-base-multilingual-cased"
        }
    }

    # Generate test data
    print("Generating test verses...")
    verses_a, verses_b = generate_test_verses(
        num_verses=100, tokens_per_verse=10)

    # Run benchmarks
    results = []

    for algo_name, config in algorithms.items():
        print(f"Benchmarking {algo_name}...")
        avg_time, success_rate, successful_runs = benchmark_algorithm(
            config, verses_a, verses_b, algo_name
        )

        results.append({
            'Algorithm': algo_name,
            'Average_Time_Seconds': avg_time,
            'Success_Rate_Percent': success_rate,
            'Successful_Runs': successful_runs,
            'Total_Runs': len(verses_a)
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Format the output
    df['Average_Time_Formatted'] = df['Average_Time_Seconds'].apply(
        lambda x: f"{x:.4f}s" if not pd.isna(x) else "FAILED"
    )

    # Save detailed results
    df.to_csv('alignment_benchmark_results.csv', index=False)

    # Create summary table for the report
    summary_df = df[['Algorithm', 'Average_Time_Formatted',
                     'Success_Rate_Percent']].copy()
    summary_df['Success_Rate_Percent'] = summary_df['Success_Rate_Percent'].apply(
        lambda x: f"{x:.1f}%")
    summary_df.columns = ['Algorithm', 'Average Time', 'Success Rate']

    summary_df.to_csv('alignment_benchmark_summary.csv', index=False)

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Algorithm':<15} {'Average Time':<15} {'Success Rate':<15}")
    print("-"*60)

    for _, row in df.iterrows():
        time_str = f"{row['Average_Time_Seconds']:.4f}s" if not pd.isna(
            row['Average_Time_Seconds']) else "FAILED"
        print(
            f"{row['Algorithm']:<15} {time_str:<15} {row['Success_Rate_Percent']:.1f}%")

    # Find fastest algorithm
    valid_results = df[df['Average_Time_Seconds'].notna()]
    if not valid_results.empty:
        fastest = valid_results.loc[valid_results['Average_Time_Seconds'].idxmin(
        )]
        print(
            f"\nFastest algorithm: {fastest['Algorithm']} ({fastest['Average_Time_Seconds']:.4f}s)")

    return df


if __name__ == "__main__":
    # Run the benchmark
    results_df = run_benchmark()

    print("\nResults saved to:")
    print("- alignment_benchmark_results.csv (detailed results)")
    print("- alignment_benchmark_summary.csv (summary table)")
