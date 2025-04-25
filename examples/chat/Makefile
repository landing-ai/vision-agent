.PHONY: setup run

setup:
	pip install -r requirements.txt
	cd chat-app && make && cd ..

run:
	npm install -g concurrently
	concurrently "bash ./run.sh" "cd chat-app && npm run dev" &
	@gnome-open http://localhost:3000 || xdg-open http://localhost:3000 || open http://localhost:3000 &

# Note: doesn't kill port, need to figure out how to.
# kill -9 $(lsof -i TCP:3010 | grep LISTEN | awk '{print $2}')