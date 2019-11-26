struct results{
    1:list<double> probs;
    2:list<double> preds;
    3:double time;
}

service Example {
    results make_prediction(1:binary arr_bytes)
    void instantiate_model()
}

