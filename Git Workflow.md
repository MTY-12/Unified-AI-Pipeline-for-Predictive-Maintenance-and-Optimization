# Branch-Based Workflow for Unified AI Pipeline 🌲

## 1. Main Branch Setup
- Ensure you are on the `main` branch and it is up to date:
  ```bash
  git checkout main
  git pull origin main
  ```

---

## 2. Predictive Maintenance
### Steps:
#### Create Branch
```bash
git checkout -b predictive-maintenance
```
#### Add Code, Test, Commit
```bash
git add .
git commit -m "Implement Predictive Maintenance with Neural Network and Decision Tree"
```
#### Push Branch
```bash
git push -u origin predictive-maintenance
```
#### Merge into Main
- Switch to `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Merge Branch:
  ```bash
  git merge predictive-maintenance
  ```
- Push Changes:
  ```bash
  git push origin main
  ```
#### Delete Branch
- Locally:
  ```bash
  git branch -d predictive-maintenance
  ```
- Remotely:
  ```bash
  git push origin --delete predictive-maintenance
  ```

---

## 3. Tool Wear Prediction
### Steps:
#### Create Branch
```bash
git checkout -b tool-wear-prediction
```
#### Add Code, Test, Commit
```bash
git add .
git commit -m "Develop regression model for tool wear prediction"
```
#### Push Branch
```bash
git push -u origin tool-wear-prediction
```
#### Merge into Main
- Switch to `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Merge Branch:
  ```bash
  git merge tool-wear-prediction
  ```
- Push Changes:
  ```bash
  git push origin main
  ```
#### Delete Branch
- Locally:
  ```bash
  git branch -d tool-wear-prediction
  ```
- Remotely:
  ```bash
  git push origin --delete tool-wear-prediction
  ```

---

## 4. RUL Estimation
### Steps:
#### Create Branch
```bash
git checkout -b rul-estimation
```
#### Add Code, Test, Commit
```bash
git add .
git commit -m "Implement LSTM and Transformer models for RUL Estimation"
```
#### Push Branch
```bash
git push -u origin rul-estimation
```
#### Merge into Main
- Switch to `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Merge Branch:
  ```bash
  git merge rul-estimation
  ```
- Push Changes:
  ```bash
  git push origin main
  ```
#### Delete Branch
- Locally:
  ```bash
  git branch -d rul-estimation
  ```
- Remotely:
  ```bash
  git push origin --delete rul-estimation
  ```

---

## 5. Process Optimization
### Steps:
#### Create Branch
```bash
git checkout -b process-optimization
```
#### Add Code, Test, Commit
```bash
git add .
git commit -m "Optimize processes using RUL and tool wear data"
```
#### Push Branch
```bash
git push -u origin process-optimization
```
#### Merge into Main
- Switch to `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Merge Branch:
  ```bash
  git merge process-optimization
  ```
- Push Changes:
  ```bash
  git push origin main
  ```
#### Delete Branch
- Locally:
  ```bash
  git branch -d process-optimization
  ```
- Remotely:
  ```bash
  git push origin --delete process-optimization
  ```

---

## 6. Bottleneck Detection
### Steps:
#### Create Branch
```bash
git checkout -b bottleneck-detection
```
#### Add Code, Test, Commit
```bash
git add .
git commit -m "Develop clustering model to detect bottlenecks in manufacturing"
```
#### Push Branch
```bash
git push -u origin bottleneck-detection
```
#### Merge into Main
- Switch to `main`:
  ```bash
  git checkout main
  git pull origin main
  ```
- Merge Branch:
  ```bash
  git merge bottleneck-detection
  ```
- Push Changes:
  ```bash
  git push origin main
  ```
#### Delete Branch
- Locally:
  ```bash
  git branch -d bottleneck-detection
  ```
- Remotely:
  ```bash
  git push origin --delete bottleneck-detection
  ```

---

## 7. Additional Git Commands Used
- **Check Current Branch**:
  ```bash
  git branch
  ```
- **Check Remote URL**:
  ```bash
  git remote -v
  ```
- **Change Remote URL**:
  ```bash
  git remote set-url origin <new-repository-url>
  ```
- **Remove Remote**:
  ```bash
  git remote remove origin
  ```
- **Initialize Repository**:
  ```bash
  git init
  ```
- **Clone Repository**:
  ```bash
  git clone <repository-url>
  ```

---

## Summary 🌟
- **Always create a branch** for each topic or feature.
- **Regularly commit and push** changes to avoid losing work.
- **Merge completed work** into the `main` branch.
- **Delete branches** after merging to keep the repository clean.

This workflow ensures organized and efficient progress across different project tasks!
