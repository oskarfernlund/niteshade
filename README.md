niteshade
=========

**niteshade** is a Python library for simulating data poisoning attack and 
defence strategies against online machine learning systems.

Releasing
---------

Releases are published automatically when a tag is pushed to GitHub.

    # Set next version number
    export RELEASE=x.x.x

    # Create tags
    git commit --allow-empty -m "Release $RELEASE"
    git tag -a $RELEASE -m "Version $RELEASE"

    # Push
    git push upstream --tags