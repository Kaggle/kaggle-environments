#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

npm i --prefix $DIR
npm run build --prefix $DIR
