"""Run image-gen commands via `python -m image_gen <command>`."""

import sys

COMMANDS = {
    "image-gen": "image_gen.gen",
    "image-gen-anim": "image_gen.gen_anim",
    "i2i-gen": "image_gen.i2i",
    "i2i-gen-anim": "image_gen.i2i_anim",
    "remove-object": "image_gen.remove",
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m image_gen <command> [args...]")
        print()
        print("Commands:")
        for name, mod in COMMANDS.items():
            print(f"  {name:20s}  ({mod})")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    sys.argv = [cmd] + sys.argv[2:]
    import importlib
    mod = importlib.import_module(COMMANDS[cmd])
    mod.main()

if __name__ == "__main__":
    main()
