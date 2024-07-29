# Vision Agent Chat Application

The Vision Agent chat appliction allows you to have conversations with the agent system
to accomplish a wider variety of tasks.

## Get Started
To get started first install the requirements by running the following command:
```bash
pip install -r requirements.txt
```

There are two environment variables you must set, the first is `WORKSPACE` which is
where the agent will look for and write files to:
```bash
export WORKSPACE=/path/to/your/workspace
```

The second is `ZMQ_PORT`, this is how the agent collects logs from subprocesses it runs
for writing code:
```bash
export ZMQ_PORT=5555
```

Finally you can launch the app with the following command:
```bash
streamlit run app.py
```

You can upload an image to your workspace in the right column first tab, then ask the
agent to do a task, (be sure to include which image you want it to use for testing) for
example:
```
Can you count the number of people in this image? Use image.jpg for testing.
```

## Layout
The are two columns, left and right, each with two tabs.

`Chat` the left column first tab is where you can chat with Vision Agent. It can answer
your questions and execute python code on your behalf. Note if you ask it to generate
vision code it may take awhile to run.

`Code Execution Logs` the left column second tab is where you will see intermediate logs
when Vision Agent is generating vision code. Because code generation can take some
time, you can monitor this tab to see what the agent is doing.

`File Browser` the right column first tab is where you can see the files in your
workspace.

`Code Editor` the right column second tab is where you can examine code files the agent
has written. You can also modify the code and save it in case the code is incorrect.
