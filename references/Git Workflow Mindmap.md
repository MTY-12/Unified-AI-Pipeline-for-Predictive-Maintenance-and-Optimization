# Git Workflow Mindmap

- **Git Workflow**
  - **1. Start a New Branch**
    - `git checkout -b branch-name` → Create and switch to a new branch.
    - Use a meaningful name, e.g., `feature-task-name` or `bugfix-task-name`.
  - **2. Work on the Branch**
    - **Add Changes**
      - Modify or create files.
      - Check changes:
        - `git status` → See which files are modified.
      - Stage changes:
        - `git add .` → Stage all changes.
        - `git add file-name` → Stage specific files.
    - **Commit Changes**
      - Save your progress locally:
        - `git commit -m "Your descriptive commit message"`
    - Repeat this process (add/commit) as you work on the branch.
  - **3. Push Branch to GitHub**
    - Push the branch to remote:
      - `git push -u origin branch-name`
    - This creates the branch in the remote repository if it doesn’t exist.
  - **4. Merge Branch into `main`**
    - **Prepare for Merging**
      - Switch to the main branch:
        - `git checkout main`
      - Pull the latest changes from the remote:
        - `git pull`
    - **Merge the Branch**
      - Merge your branch into `main`:
        - `git merge branch-name`
    - **Delete the Branch (Optional)**
      - Locally:
        - `git branch -d branch-name`
      - Remotely:
        - `git push origin --delete branch-name`
  - **5. Starting a New Feature/Task**
    - Always create a new branch for a new task.
    - Example branch types:
      - `feature-task-name` for new features.
      - `bugfix-task-name` for bug fixes.
      - `experiment-task-name` for experiments.





<!-- Step 1: Create a New Branch
git checkout -b feature-preprocessing
Step 2: Stage and Commit Changes
  Add the new script to Git:
  Commit the changes with a descriptive message:
git add src/preprocess.py
git commit -m "Add initial preprocessing script for data cleaning and feature engineering"

Step 5: Push the Branch to GitHub
git push -u origin feature-preprocessing

Verify on GitHub:

Open your repository on GitHub.
You should see the new branch feature-preprocessing and the preprocess.py script. -->





<!-- ### **Steps to Fix the Issue**
1. **Check for Running Git Processes**
   - Run this command to check if any Git processes are still running:
     ```bash
     ps aux | grep git
     ```
   - If you see a process related to Git, terminate it by noting its PID and running:
     ```bash
     kill -9 PID
     ```

2. **Manually Remove the Lock File**
   - Navigate to the Git repository and delete the lock file:
     ```bash
     rm -f .git/index.lock
     ```
   - This removes the `.git/index.lock` file, which was preventing Git operations.

3. **Retry the `git add` Command**
   - Once the lock file is removed, retry staging your changes:
     ```bash
     git add .
     ```

4. **Check for Existing Commit**
   - If the error occurred while you were committing, check if the commit succeeded:
     ```bash
     git log
     ```
   - If the commit exists, you don't need to redo it. Otherwise, retry committing:
     ```bash
     git commit -m "Your commit message"
     ```

5. **Push the Changes**
   - Once the `git add` and `git commit` steps are successful, push your changes:
     ```bash
     git push
     ```

---

### **Preventing This Issue**
- Avoid interrupting Git commands, especially during `add`, `commit`, or `merge`.
- If you're editing commit messages, ensure the editor is properly closed before running other Git commands.

Let me know if you encounter further issues! -->