"""Repository convenience entrypoint; installed equivalent: resumeanalysis prepare."""

import sys

from resumeanalysis.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "prepare")
    main()
