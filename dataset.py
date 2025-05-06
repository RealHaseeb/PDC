import random

def convert_to_weighted_edge_list(input_file, output_file, weight_min=1, weight_max=10):
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            # Write header to output file
            f_out.write("# Directed and weighted edge list from Slashdot0811.txt\n")
            f_out.write("# FromNodeId ToNodeId Weight\n")
            
            # Process input file
            for line in f_in:
                # Skip header lines
                if line.startswith('#') or line.startswith('FromNodeId'):
                    continue
                
                # Parse edge
                try:
                    from_node, to_node = map(int, line.strip().split())
                    # Generate random integer weight
                    weight = random.randint(weight_min, weight_max)
                    # Write weighted edge
                    f_out.write(f"{from_node} {to_node} {weight}\n")
                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} (Error: {e})")
                    continue
        
        print(f"Weighted edge list written to {output_file} with random integer weights between {weight_min} and {weight_max}")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    input_file = "Slashdot0811.txt"
    output_file = "weighted_edge_list.txt"
    convert_to_weighted_edge_list(input_file, output_file)