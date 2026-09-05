# 🛡️ Security Policy

The **`medical-science`** project takes the security of its humanoid robotics simulation code, scientific pipelines, and cloud training infrastructure seriously. This document outlines our security commitment, supported versions, and procedure for responsibly reporting security vulnerabilities.

---

## 📦 Supported Versions

Security patches and dependency updates are actively applied to the following branch releases:

| Version / Branch | Supported | Notes |
| :---: | :---: | :--- |
| `main` (Latest) | :white_check_mark: | Actively maintained with continuous Dependabot scanning and vulnerability updates. |
| `< 1.0.0` releases | :x: | Legacy development snapshots. Please upgrade to latest `main`. |

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability in this project, **please do not open a public issue**. Publicly disclosing a vulnerability can endanger projects and cloud environments utilizing these pipelines.

Instead, please report security concerns through one of the following channels:

### 1. GitHub Private Vulnerability Reporting (Recommended)

You can report vulnerabilities privately and directly via GitHub:

1. Navigate to the [Security Tab](https://github.com/tranvanmanh9325/medical-science/security) of this repository.
2. Click on **"Report a vulnerability"** under the Advisories section.
3. Provide a detailed summary, steps to reproduce, and any proof-of-concept (PoC) code.

### 2. Direct Maintainer Communication

- Contact the lead maintainer directly via GitHub: [@tranvanmanh9325](https://github.com/tranvanmanh9325)
- Please include:
  - Description of the issue (e.g., buffer overflow in C-bindings, credential leakage, remote code execution risks).
  - Affected components (`main.py`, `training/`, cloud deployment scripts).
  - Potential mitigation or proposed patch if available.

---

## ⏱️ Response Timeline

- **Initial Response:** Within **48 hours** of report receipt.
- **Vulnerability Assessment & Triage:** Within **5 business days**.
- **Fix Release:** Security patches will be committed to `main` as high priority once verified.
- **Public Disclosure:** Following coordinated disclosure guidelines after the fix has been pushed.

---

## 🔐 Security Best Practices for Users & Contributors

1. **Credential Hygiene:** Never commit private API tokens or credentials (e.g., `gpu/kaggle.json`, Google Cloud service accounts, Colab tokens) to git history. Keep them listed in `.gitignore`.
2. **GPU Memory & Process Isolation:** When developing custom render loops or MuJoCo hooks, always ensure process termination handlers release OpenGL/CUDA memory contexts to prevent hardware exhaustion.
3. **Dependency Integrity:** Regularly audit installed Python packages using `pip-audit` or Dependabot alerts.
