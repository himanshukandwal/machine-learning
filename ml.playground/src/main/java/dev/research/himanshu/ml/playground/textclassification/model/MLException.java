package dev.research.himanshu.ml.playground.textclassification.model;

/**
 * Machine learning exception.
 * 
 * @author Himanshu Kandwal
 */
public class MLException extends Exception {

	private static final long serialVersionUID = 1L;
	
	private static String ML_EXCEPTION = "[ML Exception] ";
	
	public MLException() {
		super();
	}

	public MLException(String message) {
		super(ML_EXCEPTION + message);
	}

	public MLException(Throwable cause) {
		super(cause);
	}

	public MLException(String message, Throwable cause) {
		super(ML_EXCEPTION + message, cause);
		// TODO Auto-generated constructor stub
	}

	public MLException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(ML_EXCEPTION + message, cause, enableSuppression, writableStackTrace);
	}

}
