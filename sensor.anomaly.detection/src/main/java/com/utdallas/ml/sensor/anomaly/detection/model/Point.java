package com.utdallas.ml.sensor.anomaly.detection.model;

/**
 * A point class.
 * 
 *  @author (Anirudh KV and Himanshu Kandwal)
 * 
 */
public class Point {

	private Integer x1;
	private Integer x2;
	private Integer x3;
	
	public Point(Integer x1, Integer x2) {
		this.x1 = x1;
		this.x2 = x2;
	}
	
	public Point(Integer x1, Integer x2, Integer x3) {
		this(x1, x2);
		this.x3 = x3;
	}
	
	public Integer getX1() {
		return x1;
	}
	
	public void setX1(Integer x1) {
		this.x1 = x1;
	}
	
	public Integer getX2() {
		return x2;
	}
	
	public void setX2(Integer x2) {
		this.x2 = x2;
	}
	
	public Integer getX3() {
		return (x3 == null ? 0 : x3);
	}
	
	public void setX3(Integer x3) {
		this.x3 = x3;
	}
	
}
