"""
Code of Conduct (coc.py)
------------------------
This module holds the Code of Conduct for contributors, users, and collaborators
working on this project. Importing this file gives access to the text as a string
or via helper functions.

Usage:
    from coc import get_coc
    print(get_coc())
"""

C_OC = """
# Code of Conduct

## 1. Purpose
Our community is built on trust, respect, and integrity. This Code of Conduct
establishes expectations for behavior and ensures a welcoming, professional,
and safe environment for all contributors and users.

## 2. Expected Behavior
- Be respectful, inclusive, and considerate in all interactions.
- Provide and accept constructive feedback gracefully.
- Respect differing viewpoints and experiences.
- Collaborate with empathy and professionalism.

## 3. Unacceptable Behavior
- Harassment, abuse, discrimination, or personal attacks.
- Use of offensive language, slurs, or derogatory remarks.
- Sharing sensitive information without consent.
- Disruptive, unprofessional, or intentionally harmful conduct.

## 4. Responsibilities
- Project maintainers are responsible for clarifying standards and enforcing them.
- Maintainers may remove, edit, or reject contributions not aligned with this Code.
- All participants are expected to uphold these standards consistently.

## 5. Enforcement
- Instances of unacceptable behavior may be reported confidentially.
- Reports will be reviewed promptly and fairly.
- Consequences may include warnings, temporary bans, or permanent removal.

## 6. Scope
This Code applies to:
- All project spaces (code, docs, issues, discussions).
- Public or private communications related to the project.
- In-person or virtual events under the project’s banner.

## 7. Attribution
This Code of Conduct is adapted from the Contributor Covenant (v2.1) and other
industry-standard policies.

---

By participating in this project, you agree to abide by this Code of Conduct.
"""

def get_coc() -> str:
    """Return the Code of Conduct as a string."""
    return C_OC

if __name__ == "__main__":
    print(get_coc())