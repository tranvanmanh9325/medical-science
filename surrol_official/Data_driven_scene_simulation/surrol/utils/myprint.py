colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"
    }

def print_c(text:str, color):
    # Get the color code, default to reset if color not found
    color_code = colors.get(color.lower(), colors["reset"])
    # Print the colored text
    print(f"{color_code}{text}{colors['reset']}")