# coding=utf-8

# from worker.meta import add_prediction
# from worker.brain import train_model
from worker.scrape import recrawl

if __name__ == "__main__":
    # print(train_model(6, (9,)))
    print(recrawl.delay().get())
