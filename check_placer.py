from intelligent_placer_lib import intelligent_placer
import json

intelligent_placer.DEBUG_OUTPUT = True

correct = []
with open("test_cases/correct_answers.json") as f:
    correct = json.loads(f.read())
answers = []
matched = 0

for i in range(0, 20):
    reason = intelligent_placer.check_image("test_cases/" + str(i) + ".jpg")
    predicted = (reason == True)
    answers.append(predicted)
    if (correct[i] == predicted):
        print("Test PASSED. No:", i, "Predicted:", predicted, "Really:", correct[i], end = " ")
        if (predicted == False):
            print("Reason:", reason)
        else:
            print("")
        matched += 1
    else:
        print("Test FAILED. No:", i, "Predicted:", predicted, "Really:", correct[i], end = " ")
        if (predicted == False):
            print("Reason:", reason)
        else:
            print("")
print("Success rate:", matched / len(answers) * 100, "%")