#!/bin/bash
# CHECK SUBMIT
# make sure there is no unsubmited changes before each running

if [ "$1" = "x" ]; then
  echo "xxxxx"
else
  echo "error"
fi

current_sha=$(git rev-parse --short HEAD)
uncommitted_changes=$(git -c color.status=always status --porcelain)
echo -e "\e[33mSHA: $current_sha\e[0m"
if [[ $uncommitted_changes ]]; then
  echo -e "\e[31mError: There are some unsubmitted changes\e[0m"
  echo -e "\e[32m======================== UNSUBMITTED CHANGES ========================\e[0m"
  echo "$uncommitted_changes"
  echo -e "\e[32m=====================================================================\e[0m"
  exit 1
fi


