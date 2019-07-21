/* Copyright 2019 Bisonai Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.bisonai.recognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.view.MotionEvent;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.ArrayList;

public class PaintView extends View {

    private static int BRUSH_SIZE = 80;
    private static final float TOUCH_TOLERANCE = 4;
    private static final int DEFAULT_BG_COLOR = Color.WHITE;
    private static final int DEFAULT_COLOR = Color.BLACK;
    private float mX, mY;
    private Path mPath;
    private Paint mPaint;
    private ArrayList<FingerPath> paths = new ArrayList<>();
    private int currentColor = DEFAULT_COLOR;
    private int backgroundColor = DEFAULT_BG_COLOR;
    private int strokeWidth = BRUSH_SIZE;
    private Bitmap mBitmap;
    private Canvas mCanvas;
    private TextView labelDrawASingleDigit;
    private LinearLayout classAndProbabilityView;
    private Integer paintSize;
    private Integer paintBorder = 10;
    private Utils utils = new Utils();

    public PaintView(Context context) {
        this(context, null);
    }

    public PaintView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(DEFAULT_COLOR);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setXfermode(null);
        mPaint.setAlpha(0xff);
    }

    public void init(DisplayMetrics metrics, TextView labelDrawASingleDigit, LinearLayout classAndProbabilityView) {
        this.labelDrawASingleDigit = labelDrawASingleDigit;
        this.classAndProbabilityView = classAndProbabilityView;

        int height = metrics.heightPixels;
        int width = metrics.widthPixels;

        if (height <= width) {
            paintSize = height;
        }
        else {
            paintSize = width;
        }

        int size = paintSize - (2 * paintBorder);

        getLayoutParams().height = size;
        getLayoutParams().width = size;

        mBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        mCanvas = new Canvas(mBitmap);
    }

    public void clear() {
        labelDrawASingleDigit.setVisibility(View.VISIBLE);
        utils.HideAllViews(classAndProbabilityView);

        backgroundColor = DEFAULT_BG_COLOR;
        paths.clear();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.save();
        mCanvas.drawColor(backgroundColor);

        for (FingerPath fp : paths) {
            mPaint.setColor(fp.color);
            mPaint.setStrokeWidth(fp.strokeWidth);
            mPaint.setMaskFilter(null);
            mCanvas.drawPath(fp.path, mPaint);
        }

        int size = paintSize - (3 * paintBorder);
        Rect rectangle = new Rect(paintBorder, paintBorder, size, size);
        canvas.drawBitmap(mBitmap, rectangle, rectangle, null);
        canvas.restore();
    }

    private void touchStart(float x, float y) {
        labelDrawASingleDigit.setVisibility(View.INVISIBLE);

        mPath = new Path();
        FingerPath fp = new FingerPath(currentColor, strokeWidth, mPath);
        paths.add(fp);

        mPath.reset();
        mPath.moveTo(x, y);
        mX = x;
        mY = y;
    }

    private void touchMove(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);

        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            mPath.quadTo(mX, mY, (x + mX) / 2, (y + mY) / 2);
            mX = x;
            mY = y;
        }
    }

    private void touchUp() {
        mPath.lineTo(mX, mY);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch(event.getAction()) {
            case MotionEvent.ACTION_DOWN :
                touchStart(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE :
                touchMove(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP :
                touchUp();
                invalidate();
                break;
        }

        return true;
    }

    public Bitmap getBitmap() {
        return mBitmap;
    }
}
