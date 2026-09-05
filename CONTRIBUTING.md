# 🤝 Contributing to Medical-Science

Thank you for your interest in contributing to the **`medical-science`** platform! We welcome contributions from roboticists, biomechanics researchers, AI engineers, and open-source developers worldwide.

---

## 🧭 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with respect, professionalism, and constructive feedback.

---

## 🛠️ How to Contribute

### 1. Reporting Bugs

- Before submitting a bug report, check the [existing issues](https://github.com/tranvanmanh9325/medical-science/issues) to avoid duplicates.
- When opening an issue, provide:
  - Clear, reproducible steps.
  - Your operating system and Python environment details.
  - Complete error tracebacks and console outputs.
  - Hardware specifications (CPU, GPU model, VRAM).

### 2. Suggesting Enhancements

We actively encourage proposals for:

- New biomechanical metrics (e.g., foot pressure center distributions, joint metabolic expenditure).
- New reinforcement learning environments (e.g., rough terrain locomotion, push recovery adaptations).
- Improved rendering shaders, HUD telemetry overlays, or physics solver configurations.

### 3. Submitting Pull Requests (PRs)

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Create a Feature Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Coding Standards:**
   - Adhere strictly to **PEP 8** conventions.
   - Follow the **DRY (Don't Repeat Yourself)** principle.
   - Write clear, self-explanatory code with concise English comments explaining *why* decisions were made.
   - Avoid breaking existing simulation loops (`main.py`) or training pipelines (`training/`).
4. **Local Verification:**
   - Always run the local smoke test before pushing:

     ```powershell
     python training/test_mini_train_sample.py
     ```

   - Verify that `main.py` launches and cleans up GPU resources cleanly without process leakage.
5. **Commit Message Format:**
   Commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat(...)`: New feature or capability
   - `fix(...)`: Bug fix
   - `perf(...)`: Performance optimization
   - `docs(...)`: Documentation updates
   - `refactor(...)`: Code refactoring without functional changes
6. **Open a Pull Request:** Submit your PR against the `main` branch with a clear description of changes.

---

## 👥 Contributors & Maintainers

We gratefully acknowledge the contributions of our project team and the open-source community. As this project expands, all contributors will be recognized in the contributor grid below.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table align="center">
  <tbody>
    <tr>
      <td align="center" valign="top" width="260">
        <a href="https://github.com/tranvanmanh9325">
          <img src="https://github.com/tranvanmanh9325.png?size=150" width="120" height="120" style="border-radius: 50%; max-width: 100%;" alt="Trần Văn Mạnh" />
          <br />
          <br />
          <b>Trần Văn Mạnh</b>
        </a>
        <br />
        <a href="https://github.com/tranvanmanh9325"><sub><b>@tranvanmanh9325</b></sub></a>
        <br />
        <br />
        <small><b>Project Lead & System Architect</b></small>
        <br />
        <small><i>Hanoi University of Science & Technology</i></small>
        <br />
        <small>(Đại học Bách Khoa Hà Nội)</small>
        <br />
        <br />
        <a href="https://github.com/tranvanmanh9325/medical-science/commits?author=tranvanmanh9325" title="Code & Architecture">💻</a>
        <a href="https://github.com/tranvanmanh9325/medical-science/tree/main/docx" title="Scientific Research">🔬</a>
        <a href="https://github.com/tranvanmanh9325/medical-science/tree/main/training" title="Reinforcement Learning Pipelines">🧠</a>
        <a href="https://github.com/tranvanmanh9325/medical-science/tree/main/docx" title="Scientific Documentation">📖</a>
        <a href="https://github.com/tranvanmanh9325/medical-science" title="Maintenance & Infrastructure">🛠️</a>
        <a href="https://github.com/tranvanmanh9325" title="Project Founder">👑</a>
        <br />
        <br />
        <a href="https://github.com/tranvanmanh9325"><img src="https://img.shields.io/github/followers/tranvanmanh9325?label=Follow%20%40tranvanmanh9325&style=social" alt="Follow on GitHub" /></a>
      </td>
      <!-- Additional contributors will be added as new <td> cells to this grid -->
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

### 🌟 Becoming a Contributor

We warmly welcome researchers, engineers, and open-source enthusiasts to join the `medical-science` initiative! When your pull request is merged, your profile will be added to the contributor grid above.

- 📖 Review the [Contribution Workflow](#-how-to-contribute) above.
- 💡 Submit bug reports or feature proposals on our [Issue Tracker](https://github.com/tranvanmanh9325/medical-science/issues).
- 🚀 Submit a [Pull Request](https://github.com/tranvanmanh9325/medical-science/pulls) to contribute to whole-body biomechanics, reinforcement learning pipelines, or surgical simulation suites.

<p align="center">
  <a href="https://github.com/tranvanmanh9325/medical-science/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=tranvanmanh9325/medical-science" alt="Contributors" />
  </a>
</p>
