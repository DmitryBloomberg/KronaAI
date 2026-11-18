#!/bin/sh
set -eu

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Создан .env. Добавьте TELEGRAM_BOT_TOKEN и снова запустите ./start.sh"
  exit 1
fi

docker compose up --build -d
echo "KronaAI запущен. Логи: docker compose logs -f bot"
