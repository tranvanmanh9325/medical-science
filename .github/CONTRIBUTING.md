# 🤝 Contributing to Medical-Science

Thank you for your interest in contributing to the **`medical-science`** platform! We welcome contributions from roboticists, biomechanics researchers, AI engineers, and open-source developers worldwide.

---

## 🧭 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](../CODE_OF_CONDUCT.md). Please treat all contributors with respect, professionalism, and constructive feedback.

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

### 🏛️ Lead Architect & Maintainer

<table>
  <tbody>
    <tr>
      <td align="center" valign="middle" width="160">
        <a href="https://github.com/tranvanmanh9325">
          <img src="https://avatars.githubusercontent.com/u/86815257?v=4" width="100" height="100" style="border-radius: 50%; max-width: 100%;" alt="Trần Văn Mạnh" /><br />
          <b>Trần Văn Mạnh</b>
        </a><br />
        <sub><a href="https://github.com/tranvanmanh9325">@tranvanmanh9325</a></sub>
      </td>
      <td valign="top">
        <b>Project Lead, System Architect & Primary Maintainer</b><br />
        🎓 <i>Hanoi University of Science and Technology (Đại học Bách Khoa Hà Nội)</i><br />
        🔬 <b>Focus Areas:</b> Humanoid Robotics Architecture, Parallel RL Pipelines (JAX/MuJoCo MJX), Whole-Body Biomechanics Diagnostics, Surgical Robotics Teleoperation (dVRK/SurRoL).<br />
        🌐 <a href="https://github.com/tranvanmanh9325"><img src="https://img.shields.io/github/followers/tranvanmanh9325?label=Follow%20%40tranvanmanh9325&style=social" alt="Follow tranvanmanh9325" /></a>
      </td>
    </tr>
  </tbody>
</table>

### 🤝 Contribution Directory

| Contributor | Role | Focus Area |
| :---: | :---: | :--- |
| <a href="https://github.com/tranvanmanh9325"><img src="https://avatars.githubusercontent.com/u/86815257?v=4" width="48" height="48" style="border-radius: 50%;" alt="Trần Văn Mạnh" /><br />**Trần Văn Mạnh**<br />`(@tranvanmanh9325)`</a> | **Project Lead & Maintainer**<br />*(Hanoi University of Science and Technology)* | Humanoid Robotics Architecture, RL Pipelines, Biomechanics Diagnostics, dVRK Surgical Suite |
| <a href="https://github.com/tranvanmanh9325/medical-science/graphs/contributors"><img src="https://contrib.rocks/image?repo=tranvanmanh9325/medical-science" width="80" alt="Contributors" /><br />**Open Source Community**</a> | **Contributors & Researchers** | Bug reports, feature enhancements, documentation, and model validations |

We gratefully acknowledge contributions, feedback, and citations from the broader robotics and AI research communities.
