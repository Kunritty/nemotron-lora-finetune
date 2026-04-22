import logging

from langchain_ollama import ChatOllama
from registry import models, datasets

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Step 1/5: Resolving ChatOllama model (gemma4)")
    model: ChatOllama = models.get_model("gemma4")
    logger.info("Step 1/5: Model ready — %s", getattr(model, "model", model))

    logger.info("Step 2/5: Loading training dataframe")
    train_df = datasets.get_train_df()
    logger.info(
        "Step 2/5: Loaded — rows=%s, columns=%s",
        len(train_df),
        list(train_df.columns),
    )

    logger.info("Step 3/5: Sampling one random prompt from train set")
    problem = train_df.sample(1)["prompt"][0]
    answer = train_df.sample(1)["answer"][0]
    logger.info("Step 3/5: Prompt preview (%s chars): %s", len(problem), problem)
    logger.info("Step 3/5: : Answer", answer)

    logger.info("Step 4/5: Invoking model (this may take a while)")
    response = model.invoke(problem)
    logger.info("Step 4/5: Invoke finished")

    logger.info("Step 5/5: Printing response")
    print("THOUGHT PROCESS:", response.additional_kwargs.get("reasoning_content"))
    print("FINAL ANSWER:", response.content)
    logger.info("Step 5/5: Done")


if __name__ == "__main__":
    main()