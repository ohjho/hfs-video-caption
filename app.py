import gradio as gr


def greet(name):
    return "Hello " + name


# We instantiate the Textbox class
textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)

app = gr.Interface(fn=greet, inputs=textbox, outputs="text")
app.launch()
