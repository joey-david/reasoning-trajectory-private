rsync -avz --delete \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  ./ lamgate:/home/lamsade/jdavid/research/reasoning
