"""Repository convenience entrypoint; installed equivalent: resumeanalysis evaluate."""

import sys

from resumeanalysis.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
