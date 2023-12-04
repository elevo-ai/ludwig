# Instructions to checkout , build and install foresight/ludwig 0.8.2

## One time  project dev setup

Pre-requisite:  access to elevoai repository from your github account. 

In your local set up perform the following steps.

git clone git@github.com:elevo-ai/ludwig
git remote add <your alias name> git@github.com:<your fork>/ludwig
Note: Create <your fork> for the first time(once)  via GitHub web
 
git checkout -b <local  dev branch> origin/foresight-ludwig-0.8.2

For committing the pushing your changes do the following

git commit -a 
git push <your alias name> <local dev branch>

When you create PR make sure to select the branch foresight-ludwig-0.8.2 instead of master


## build instructions

Create your custom virtual environment using command python3 -m venv ludwig_foresight-dev_venv.

run command pip install  build , to install build library.

From the project root directory run the command python3 -m build --sdist , this will create a dist directory , with a tar file name ludwig-0.8.2.tar.gz as a binary output.
Rename it to foresight-ludwig-0.8.2.tar.gz.

## install instructions

If you want to install the tar file in your desired environment, you can do it  by installing the command pip install foresight-ludwig-0.8.2.tar.gz

After the changes are approved. Copy the tar file to ${TOOLSDIR}/foresight-dependfiles folder. So that package-dlearn.sh script picks up the updated tar file.




